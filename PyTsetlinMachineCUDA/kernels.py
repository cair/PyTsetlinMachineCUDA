# Copyright (c) 2020 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

code_header = """
	#include <curand_kernel.h>
	
	#define INT_SIZE 32

	#define LA_CHUNKS (((FEATURES-1)/INT_SIZE + 1))
	#define CLAUSE_CHUNKS ((CLAUSES-1)/INT_SIZE + 1)

	#if (FEATURES % 32 != 0)
	#define FILTER (~(0xffffffff << (FEATURES % INT_SIZE)))
	#else
	#define FILTER 0xffffffff
	#endif
"""

code_update = """
	extern "C"
    {
    	// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
		__device__ inline void inc(unsigned int *ta_state, int clause, int chunk, unsigned int active)
		{
			unsigned int carry, carry_next;
			int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;
			carry = active;
			for (int b = 0; b < STATE_BITS; ++b) {
				if (carry == 0)
					break;

				carry_next = ta_state[id + b] & carry; // Sets carry bits (overflow) passing on to next bit
				ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
				carry = carry_next;
			}

			if (carry > 0) {
				for (int b = 0; b < STATE_BITS; ++b) {
					ta_state[id + b] |= carry;
				}
			}   
		}

		// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
		__device__ inline void dec(unsigned int *ta_state, int clause, int chunk, unsigned int active)
		{
			unsigned int carry, carry_next;
			int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;
			carry = active;
			for (int b = 0; b < STATE_BITS; ++b) {
				if (carry == 0)
					break;
				carry_next = (~ta_state[id + b]) & carry; // Sets carry bits (overflow) passing on to next bit
				ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
				carry = carry_next;
			}

			if (carry > 0) {
				for (int b = 0; b < STATE_BITS; ++b) {
					ta_state[id + b] &= ~carry;
				}
			} 
		}

		// Update state of Tsetlin Automata team over batch
		__global__ void update(curandState *state, unsigned int *global_ta_state, unsigned char *clause_weights, int *class_sum, unsigned char *global_clause_output, int *global_clause_patch, int *X, int *y, int example)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;
			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];
			for (int batch = 0; batch < BATCH_SIZE; ++batch) {			    
				int e = (example + batch);
				if (e >= NUMBER_OF_EXAMPLES) {
					break;
				}

				// Calculate clause output first
				for (int i = index; i < CLASSES*CLAUSES; i += stride) {
					unsigned long long class_id = i / CLAUSES;
					unsigned long long clause = i % CLAUSES;
					unsigned char clause_weight = clause_weights[i];
					int output_one_patches[PATCHES];
					int output_one_patches_count;
					unsigned int *ta_state = &global_ta_state[class_id*CLAUSES*LA_CHUNKS*STATE_BITS + clause*LA_CHUNKS*STATE_BITS];
					unsigned char *clause_output = &global_clause_output[class_id*CLAUSES*NUMBER_OF_EXAMPLES + clause*NUMBER_OF_EXAMPLES];
					
					#if NEGATIVE_CLAUSES == 1
						int sign = 1 - 2*(clause & 1);
					#else
						int sign = 1;
					#endif

					// Evaluate each patch (convolution)
					output_one_patches_count = 0;
					for (int patch = 0; patch < PATCHES; ++patch) {
						int patch_clause_output = 1;
						for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
							if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
								patch_clause_output = 0;
								break;
							}
						}

						if (((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + LA_CHUNKS - 1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER))) {
							patch_clause_output = 0;
						}

						if (patch_clause_output) {
							output_one_patches[output_one_patches_count] = patch;
							output_one_patches_count++;
						}
					}
				
					unsigned char clause_output_new;
					if (output_one_patches_count > 0) {
						clause_output_new = clause_weight;
						int patch_id = curand(&localState) % output_one_patches_count;
						global_clause_patch[class_id*CLAUSES + clause] = output_one_patches[patch_id];
					} else {
						clause_output_new = 0;
						global_clause_patch[class_id*CLAUSES + clause] = -1;
					}

					if (clause_output_new != clause_output[e]) {
						atomicAdd(&class_sum[class_id*NUMBER_OF_EXAMPLES + e], sign*(clause_output_new - clause_output[e]));
						clause_output[e] = clause_output_new;
					}
				}

				// Provide feedback to Tsetlin Automata
				for (int i = index; i < CLASSES*CLAUSES; i += stride) {
					unsigned long long class_id = i / CLAUSES;
					unsigned long long clause = i % CLAUSES;
					unsigned char clause_weight = clause_weights[i];
					unsigned int *ta_state = &global_ta_state[class_id*CLAUSES*LA_CHUNKS*STATE_BITS + clause*LA_CHUNKS*STATE_BITS];
					unsigned char *clause_output = &global_clause_output[class_id*CLAUSES*NUMBER_OF_EXAMPLES + clause*NUMBER_OF_EXAMPLES];
					#if NEGATIVE_CLAUSES == 1
						int sign = 1 - 2*(clause & 1);
					#else
						int sign = 1;
					#endif
				
					int clause_patch = global_clause_patch[class_id*CLAUSES + clause];
					int local_class_sum = class_sum[class_id*NUMBER_OF_EXAMPLES + e];
					if (local_class_sum > THRESHOLD) {
						local_class_sum = THRESHOLD;
					} else if (local_class_sum < -THRESHOLD) {
						local_class_sum = -THRESHOLD;
					}
				
					int target = 1 - 2*(local_class_sum > y[e*CLASSES + class_id]);
					if (target == -1 && curand_uniform(&localState) > 1.0/max(1, CLASSES-1)) {
						continue;
					}
				
					int absolute_prediction_error = abs(y[e*CLASSES + class_id] - local_class_sum);
					if (curand_uniform(&localState) <= 1.0*absolute_prediction_error/(2*THRESHOLD)) {
						if (target*sign > 0) {
							#if MAX_WEIGHT > 1
								if (clause_output[e] && clause_weight < MAX_WEIGHT) {
								clause_weight++;
								}
							#endif

							// Type I Feedback
							for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
								// Generate random bit values
								unsigned int la_feedback = 0;
								for (int b = 0; b < INT_SIZE; ++b) {
									if (curand_uniform(&localState) <= 1.0/S) {
										la_feedback |= (1 << b);
									}
								}

								if (clause_output[e]) {
									#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
										inc(ta_state, 0, la_chunk, X[e*(LA_CHUNKS*PATCHES) + clause_patch*LA_CHUNKS + la_chunk]);
									#else
										inc(ta_state, 0, la_chunk, X[e*(LA_CHUNKS*PATCHES) + clause_patch*LA_CHUNKS + la_chunk] & (~la_feedback));
									#endif

									dec(ta_state, 0, la_chunk, (~X[e*(LA_CHUNKS*PATCHES) + clause_patch*LA_CHUNKS + la_chunk]) & la_feedback);
								} else {
									dec(ta_state, 0, la_chunk, la_feedback);
								}
							}
						} else if (target*sign < 0 && clause_output[e]) {
							// Type II Feedback
							#if MAX_WEIGHT > 1
								if (clause_weight > 1) {
									clause_weight--;
								}
							#endif

							for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
								inc(ta_state, 0, la_chunk, (~X[e*(LA_CHUNKS*PATCHES) + clause_patch*LA_CHUNKS + la_chunk]) & (~ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]));
							}
						}
					}

					clause_weights[i] = clause_weight;
				}
			}
			state[index] = localState;
		}
    }
"""

code_evaluate = """
	extern "C"
    {
		// Evaluate examples
		__global__ void evaluate(unsigned int *global_ta_state, unsigned char *clause_weights, int *class_sum, int *X)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int i = index; i < CLASSES*CLAUSES; i += stride) {
				unsigned long long class_id = i / CLAUSES;
				unsigned long long clause = i % CLAUSES;

				unsigned char clause_weight = clause_weights[i];

				unsigned int *ta_state = &global_ta_state[class_id*CLAUSES*LA_CHUNKS*STATE_BITS + clause*LA_CHUNKS*STATE_BITS];

				int all_exclude = 1;
				for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
					if (ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] > 0) {
						all_exclude = 0;
						break;
					}
				}

				if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER) > 0) {
					all_exclude = 0;
				}

				if (all_exclude) {
					continue;
				}

				#if NEGATIVE_CLAUSES == 1
					int sign = 1 - 2*(clause & 1);
				#else
					int sign = 1;
				#endif

				for (int e = 0; e < NUMBER_OF_EXAMPLES; ++e) {
					unsigned char clause_output;
					for (int patch = 0; patch < PATCHES; ++patch) {
						clause_output = 1;
						for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
							if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
								clause_output = 0;
								break;
							}
						}

						if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + LA_CHUNKS-1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
							clause_output = 0;
						}

						if (clause_output) {
							break;
						}
					}

					if (clause_output) {
						atomicAdd(&class_sum[class_id*NUMBER_OF_EXAMPLES + e], sign*clause_weight);
					}
				}
			}
		}
	}
"""

code_prepare = """
	extern "C"
    {
		__global__ void prepare(unsigned int *global_ta_state, unsigned char *clause_weights, unsigned char *global_clause_output, int *class_sum)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int i = index; i < CLASSES*CLAUSES; i += stride) {
				unsigned long long class_id = i / CLAUSES;
				unsigned long long clause = i % CLAUSES;

				unsigned char *clause_output = &global_clause_output[class_id*CLAUSES*NUMBER_OF_EXAMPLES + clause*NUMBER_OF_EXAMPLES];

				clause_weights[i] = 1;

				unsigned int *ta_state = &global_ta_state[class_id*CLAUSES*LA_CHUNKS*STATE_BITS + clause*LA_CHUNKS*STATE_BITS];

				for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
					for (int b = 0; b < STATE_BITS-1; ++b) {
						ta_state[la_chunk*STATE_BITS + b] = ~0;
					}
					ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] = 0;
				}

				for (int example = 0; example < NUMBER_OF_EXAMPLES; ++example) {
					clause_output[example] = 1;
				}

				for (int example = 0; example < NUMBER_OF_EXAMPLES; ++example) {
					#if NEGATIVE_CLAUSES == 1
						class_sum[class_id*NUMBER_OF_EXAMPLES + example] = 0;
					#else
						class_sum[class_id*NUMBER_OF_EXAMPLES + example] = CLAUSES;
					#endif
				}
			}
		}
	}
"""

code_encode = """
	#include <curand_kernel.h>

	extern "C"
    {
		__global__ void prepare_encode(unsigned int *X, unsigned int *encoded_X, int number_of_examples, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int number_of_features = class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
			int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

			int number_of_ta_chunks;
			if (append_negated) {
				number_of_ta_chunks= (((2*number_of_features-1)/32 + 1));
			} else {
				number_of_ta_chunks= (((number_of_features-1)/32 + 1));
			}

			for (int i = index; i < number_of_examples * number_of_patches * number_of_ta_chunks; i += stride) {
				encoded_X[i] = 0;
			}
		}
	
		__global__ void encode(unsigned int *X, unsigned int *encoded_X, int number_of_examples, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int global_number_of_features = dim_x * dim_y * dim_z;
			int number_of_features = class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
			int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

			int number_of_ta_chunks;
			if (append_negated) {
				number_of_ta_chunks= (((2*number_of_features-1)/32 + 1));
			} else {
				number_of_ta_chunks= (((number_of_features-1)/32 + 1));
			}

			unsigned int *Xi;
			unsigned int *encoded_Xi;

			unsigned int input_step_size = global_number_of_features;

			for (int i = index; i < number_of_examples; i += stride) {
				unsigned int encoded_pos = i * number_of_patches * number_of_ta_chunks;
				unsigned int input_pos = i * input_step_size;

				int patch_nr = 0;
				// Produce the patches of the current image
				for (int y = 0; y < dim_y - patch_dim_y + 1; ++y) {
					for (int x = 0; x < dim_x - patch_dim_x + 1; ++x) {
						Xi = &X[input_pos];
						encoded_Xi = &encoded_X[encoded_pos];

						// Encode class into feature vector 
						for (int class_feature = 0; class_feature < class_features; ++class_feature) {

							int chunk_nr = (class_feature + number_of_features) / 32;
							int chunk_pos = (class_feature + number_of_features) % 32;
							encoded_Xi[chunk_nr] |= (1 << chunk_pos);
						}

						// Encode y coordinate of patch into feature vector 
						for (int y_threshold = 0; y_threshold < dim_y - patch_dim_y; ++y_threshold) {
							int patch_pos = class_features + y_threshold;

							if (y > y_threshold) {
								int chunk_nr = patch_pos / 32;
								int chunk_pos = patch_pos % 32;
								encoded_Xi[chunk_nr] |= (1 << chunk_pos);
							} else if (append_negated) {
								int chunk_nr = (patch_pos + number_of_features) / 32;
								int chunk_pos = (patch_pos + number_of_features) % 32;
								encoded_Xi[chunk_nr] |= (1 << chunk_pos);
							}
						}

						// Encode x coordinate of patch into feature vector
						for (int x_threshold = 0; x_threshold < dim_x - patch_dim_x; ++x_threshold) {
							int patch_pos = class_features + (dim_y - patch_dim_y) + x_threshold;

							if (x > x_threshold) {
								int chunk_nr = patch_pos / 32;
								int chunk_pos = patch_pos % 32;

								encoded_Xi[chunk_nr] |= (1 << chunk_pos);
							} else if (append_negated) {
								int chunk_nr = (patch_pos + number_of_features) / 32;
								int chunk_pos = (patch_pos + number_of_features) % 32;
								encoded_Xi[chunk_nr] |= (1 << chunk_pos);
							}
						} 

						// Encode patch content into feature vector
						for (int p_y = 0; p_y < patch_dim_y; ++p_y) {
							for (int p_x = 0; p_x < patch_dim_x; ++p_x) {
								for (int z = 0; z < dim_z; ++z) {
									int image_pos = (y + p_y)*dim_x*dim_z + (x + p_x)*dim_z + z;
									int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;

									if (Xi[image_pos] == 1) {
										int chunk_nr = patch_pos / 32;
										int chunk_pos = patch_pos % 32;
										encoded_Xi[chunk_nr] |= (1 << chunk_pos);
									} else if (append_negated) {
										int chunk_nr = (patch_pos + number_of_features) / 32;
										int chunk_pos = (patch_pos + number_of_features) % 32;
										encoded_Xi[chunk_nr] |= (1 << chunk_pos);
									}
								}
							}
						}
						encoded_pos += number_of_ta_chunks;
						patch_nr++;
					}
				}
			}
		}
	}
"""

code_transform = """
	extern "C"
    {
		// Transform examples
		__global__ void transform(unsigned int *global_ta_state, int *X, int *transformed_X)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int i = index; i < CLASSES*CLAUSES; i += stride) {
				unsigned long long class_id = i / CLAUSES;
				unsigned long long clause = i % CLAUSES;

				unsigned int *ta_state = &global_ta_state[class_id*CLAUSES*LA_CHUNKS*STATE_BITS + clause*LA_CHUNKS*STATE_BITS];

				int all_exclude = 1;
				for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
					if (ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] > 0) {
						all_exclude = 0;
						break;
					}
				}

				if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER) > 0) {
					all_exclude = 0;
				}

				if (all_exclude) {
					for (unsigned long long e = 0; e < NUMBER_OF_EXAMPLES; ++e) {
						transformed_X[e*CLASSES*CLAUSES + i] = 0;
					}
					
					continue;
				}

				for (int e = 0; e < NUMBER_OF_EXAMPLES; ++e) {
					unsigned char clause_output;
					for (int patch = 0; patch < PATCHES; ++patch) {
						clause_output = 1;
						for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
							if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
								clause_output = 0;
								break;
							}
						}

						if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + LA_CHUNKS-1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
							clause_output = 0;
						}

						if (clause_output) {
							break;
						}
					}

					transformed_X[e*CLASSES*CLAUSES + i] = clause_output;
				}
			}
		}
	}
"""
