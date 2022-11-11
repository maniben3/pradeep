extern "C"
void init(float *t_height_map,
float *w_height_map,
float *s_height_map,
int SIZE_X,
int SIZE_Y);

extern "C"
void run_hydro_erosion(int cycles,
float t_step,
float min_tilt_angle,
float SEDIMENT_CAP,
float DISSOLVE_CONST,
float DEPOSIT_CONST,
int SIZE_X,
int SIZE_Y,
float PIPE_LENGTH,
float ADJACENT_LENGTH,
float TIME_STEP,
float MIN_TILT_ANGLE);

extern "C"
void free_mem();

extern "C"
void procedural_rain(float *water_height_map, float *rain_map, int SIZE_X, int SIZE_Y);

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <random>

// includes CUDA
#include <cuda_runtime.h>

using namespace std;

#define FLOW_RIGHT 0
#define FLOW_UP 1
#define FLOW_LEFT 2
#define FLOW_DOWN 3
#define X_VEL 0
#define Y_VEL 1
#define LEFT_CELL row, col - 1
#define RIGHT_CELL row, col + 1
#define ABOVE_CELL row - 1, col
#define BELOW_CELL row + 1, col

// CUDA API error checking macro
#define T 1024
#define M 1536
#define blockSize 1024
#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("Fatal error: %s at %s:%d\n", \
      cudaGetErrorString(error), \
      __FILE__, __LINE__); \
    exit(1); \
              }


__global__ void update_water_flow(float *water_height_map, float *water_flow_map, float *d_updated_water_flow_map, int SIZE_X, int SIZE_Y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int col = index % SIZE_X;
    int row = index / SIZE_X;

    index = row * (SIZE_X * 4) + col * 4;   // 3D index
#ifdef FIX
    if ((row >= SIZE_Y) || (col >= SIZE_X)) return;
#endif
    d_updated_water_flow_map[index + FLOW_RIGHT] = 0;
    d_updated_water_flow_map[index + FLOW_UP] = 0;
    d_updated_water_flow_map[index + FLOW_LEFT] = 0;
    d_updated_water_flow_map[index + FLOW_DOWN] = 0;

}

static float *terrain_height_map;
static float *water_height_map;
static float *sediment_height_map;

void init(float *t_height_map,
    float *w_height_map,
    float *s_height_map,
    int SIZE_X,
    int SIZE_Y)
{
    /* set vars HOST*/
    terrain_height_map = t_height_map;
    water_height_map = w_height_map;
    sediment_height_map = s_height_map;
}

void run_hydro_erosion(int cycles,
    float t_step,
    float min_tilt_angle,
    float SEDIMENT_CAP,
    float DISSOLVE_CONST,
    float DEPOSIT_CONST,
    int SIZE_X,
    int SIZE_Y,
    float PIPE_LENGTH,
    float ADJACENT_LENGTH,
    float TIME_STEP,
    float MIN_TILT_ANGLE)
{
    int numBlocks = (SIZE_X * SIZE_Y + (blockSize - 1)) / blockSize;
    int SIZE = SIZE_X * SIZE_Y * sizeof(float);

    float *d_terrain_height_map, *d_updated_terrain_height_map;
    float *d_water_height_map, *d_updated_water_height_map;
    float *d_sediment_height_map, *d_updated_sediment_height_map;

    float *d_suspended_sediment_level;
    float *d_updated_suspended_sediment_level;
    float *d_water_flow_map;
    float *d_updated_water_flow_map;
    float *d_prev_water_height_map;
    float *d_water_velocity_vec;
    float *d_rain_map;

    cudaCheck(cudaMalloc(&d_water_height_map, SIZE));
    cudaCheck(cudaMalloc(&d_updated_water_height_map, SIZE));
    cudaCheck(cudaMalloc(&d_prev_water_height_map, SIZE));
    cudaCheck(cudaMalloc(&d_water_flow_map, SIZE * 4));
    cudaCheck(cudaMalloc(&d_updated_water_flow_map, SIZE * 4)); // changing this array also changes d_terrain_height_map
    cudaCheck(cudaMalloc(&d_terrain_height_map, SIZE));
    cudaCheck(cudaMalloc(&d_updated_terrain_height_map, SIZE));
    cudaCheck(cudaMalloc(&d_sediment_height_map, SIZE));
    cudaCheck(cudaMalloc(&d_updated_sediment_height_map, SIZE));
    cudaCheck(cudaMalloc(&d_suspended_sediment_level, SIZE));
    cudaCheck(cudaMalloc(&d_updated_suspended_sediment_level, SIZE));
    cudaCheck(cudaMalloc(&d_rain_map, SIZE));
    cudaCheck(cudaMalloc(&d_water_velocity_vec, SIZE * 2));

    cudaCheck(cudaMemcpy(d_terrain_height_map, terrain_height_map, SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_water_height_map, water_height_map, SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sediment_height_map, sediment_height_map, SIZE, cudaMemcpyHostToDevice));

    cout << "init terrain_height_map" << endl;
    for (int i = 0; i < SIZE_X * SIZE_Y; i++) {
        cout << terrain_height_map[i] << ", ";
        if (i % SIZE_X == 0 && i != 0) cout << endl;
    }

    /* launch the kernel on the GPU */
    float *temp;
    while (cycles--) {
        update_water_flow << < numBlocks, blockSize >> >(d_water_height_map, d_water_flow_map, d_updated_water_flow_map, SIZE_X, SIZE_Y);
        temp = d_water_flow_map;
        d_water_flow_map = d_updated_water_flow_map;
        d_updated_water_flow_map = temp;
    }
    cudaCheck(cudaMemcpy(terrain_height_map, d_terrain_height_map, SIZE, cudaMemcpyDeviceToHost));


    cout << "updated terrain" << endl;
    for (int i = 0; i < SIZE_X * SIZE_Y; i++) {
        cout << terrain_height_map[i] << ", ";
        if (i % SIZE_X == 0 && i != 0) cout << endl;
    }
}
