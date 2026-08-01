// Streamed NZ packing (production, [N,K] -> per-16-column groups) must produce
// byte-identical output to the reference pack_fractal_nz ([K,N] -> NZ).
#include "minicpmo/tensor.h"
#include <cstdio>
#include <random>
#include <vector>
using namespace minicpmo;
int main(){
  for (auto [K,N] : std::vector<std::pair<int64_t,int64_t>>{{64,32},{256,512},{4096,1024}}) {
    std::mt19937 rng(7); std::vector<uint16_t> nk((size_t)N*K), kn((size_t)K*N);
    for(auto& v:nk) v=(uint16_t)(rng()&0xffff);
    for(int64_t n=0;n<N;n++)for(int64_t k=0;k<K;k++) kn[(size_t)k*N+n]=nk[(size_t)n*K+k];
    std::vector<uint16_t> ref((size_t)K*N);
    pack_fractal_nz(kn.data(),ref.data(),K,N);
    // streamed variant, mirroring load_matmul_weight_transposed
    std::vector<uint16_t> got((size_t)K*N); size_t ge=(size_t)K*16;
    std::vector<uint16_t> g(ge);
    for(int64_t nb=0;nb<N/16;nb++){
      std::fill(g.begin(),g.end(),(uint16_t)0);
      for(int64_t ni=0;ni<16;ni++){const uint16_t* r=&nk[(size_t)(nb*16+ni)*K];
        for(int64_t k=0;k<K;k++) g[(size_t)(k/16)*256u+(size_t)(k%16)*16u+(size_t)ni]=r[k];}
      std::copy(g.begin(),g.end(),got.begin()+(size_t)nb*ge);
    }
    size_t bad=0; for(size_t i=0;i<ref.size();i++) if(ref[i]!=got[i]) bad++;
    std::printf("K=%-5ld N=%-5ld mismatches=%zu %s\n",(long)K,(long)N,bad,bad?"FAIL":"PASS");
  }
  return 0;
}
