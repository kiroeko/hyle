int hyle_cxx_probe(const int*);
int hyle_cuda_host_probe(const int*);
extern "C" int hyle_asm_probe();
int main() {
  const int value = 7;
  return hyle_cxx_probe(&value) == 7 && hyle_cuda_host_probe(&value) == 7 &&
                 hyle_asm_probe() == 0
             ? 0
             : 1;
}
