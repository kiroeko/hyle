// One whole-batch child, exact sealed timed image and inherited policy.
// The adapter reports controls and forwards raw output; it performs no math.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {
bool forward(const std::filesystem::path &path,FILE *out) {
  FILE *in=nullptr;if(_wfopen_s(&in,path.c_str(),L"rb"))return false;
  std::array<char,16384> bytes{};
  for(;;){const auto n=std::fread(bytes.data(),1,bytes.size(),in);if(n && std::fwrite(bytes.data(),1,n,out)!=n){std::fclose(in);return false;}if(n<bytes.size())break;}
  const bool ok=!std::ferror(in);std::fclose(in);std::fflush(out);return ok;
}
}
int main() {
  SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
  const auto *cpu=std::getenv("HYLE_PERF_CALLER_CPU");
  const auto *nonce=std::getenv("HYLE_PERF_RUN_NONCE");
  if(!cpu || !nonce)return 70;
  const auto caller=std::strtoul(cpu,nullptr,10);if(caller>=64)return 70;
  DWORD_PTR process_mask=0,system_mask=0;
  const auto mask=DWORD_PTR{1}<<caller;
  if(!GetProcessAffinityMask(GetCurrentProcess(),&process_mask,&system_mask) || process_mask!=mask ||
      !SetThreadAffinityMask(GetCurrentThread(),mask) || GetCurrentProcessorNumber()!=caller)return 71;
  const auto priority=GetPriorityClass(GetCurrentProcess());if(!priority)return 72;
  std::array<wchar_t,32768> name{};const auto size=GetModuleFileNameW(nullptr,name.data(),static_cast<DWORD>(name.size()));
  if(!size || size==name.size())return 73;
  const auto directory=std::filesystem::path(name.data()).parent_path();
  const auto child=directory/L"hyle_cpu_best_of_sealed.exe";
  const auto stdout_path=directory/L"f4-batch.stdout.tmp",stderr_path=directory/L"f4-batch.stderr.tmp";
  SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
  const auto output=CreateFileW(stdout_path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&attributes,CREATE_ALWAYS,FILE_ATTRIBUTE_TEMPORARY,nullptr);
  const auto error=CreateFileW(stderr_path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&attributes,CREATE_ALWAYS,FILE_ATTRIBUTE_TEMPORARY,nullptr);
  if(output==INVALID_HANDLE_VALUE || error==INVALID_HANDLE_VALUE) {
    if(output!=INVALID_HANDLE_VALUE)CloseHandle(output);if(error!=INVALID_HANDLE_VALUE)CloseHandle(error);return 74;
  }
  STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES|STARTF_USESHOWWINDOW;startup.wShowWindow=SW_HIDE;
  startup.hStdInput=GetStdHandle(STD_INPUT_HANDLE);startup.hStdOutput=output;startup.hStdError=error;
  PROCESS_INFORMATION process{};std::wstring command=L"\""+child.wstring()+L"\"";
  const auto created=CreateProcessW(child.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_SUSPENDED|priority,nullptr,directory.c_str(),&startup,&process);
  CloseHandle(output);CloseHandle(error);if(!created)return 75;
  if(!SetProcessAffinityMask(process.hProcess,mask) || ResumeThread(process.hThread)==DWORD(-1)) {
    TerminateProcess(process.hProcess,76);CloseHandle(process.hThread);CloseHandle(process.hProcess);return 76;
  }
  DWORD_PTR child_mask=0,child_system=0;
  if(!GetProcessAffinityMask(process.hProcess,&child_mask,&child_system) || child_mask!=mask || GetPriorityClass(process.hProcess)!=priority) {
    TerminateProcess(process.hProcess,77);CloseHandle(process.hThread);CloseHandle(process.hProcess);return 77;
  }
  std::fprintf(stderr,"# CPU_BEST_OF_ADAPTER execution_order=%s allocation_order=%s affinity_topology=instrument caller_cpu=%s affinity_ok=1 run_nonce=%s one_batch=1 child_policy_verified=1\n",
      std::getenv("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",
      std::getenv("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",cpu,nonce);
  const auto waited=WaitForSingleObject(process.hProcess,INFINITE);DWORD code=78;const auto read=GetExitCodeProcess(process.hProcess,&code);
  CloseHandle(process.hThread);CloseHandle(process.hProcess);
  if(!forward(stdout_path,stdout) || !forward(stderr_path,stderr))return 79;
  DeleteFileW(stdout_path.c_str());DeleteFileW(stderr_path.c_str());
  if(waited!=WAIT_OBJECT_0 || !read)return 78;
  return static_cast<int>(code);
}
