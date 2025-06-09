#include <cstdint>

int g_thread_idx;
int g_block_dim;
int g_block_idx;
int g_lane_id;
int g_warp_id;

extern "C" int thread_idx()   { return g_thread_idx; }
extern "C" int block_dim()    { return g_block_dim; }
extern "C" int block_idx()    { return g_block_idx; }
extern "C" int lane_id()      { return g_lane_id; }
extern "C" int warp_id()      { return g_warp_id; }
