// Windows development transport. The sealed child owns all timed math; this
// process selects only prequalified CLI requests and forwards their raw tables.
#include <array>
#include <cstdio>
#include <filesystem>
#include <string>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {
struct Pair final {const wchar_t *case_id;const wchar_t *candidate;const wchar_t *peer;};
#include "p1_pairs.h"

bool forward(const std::filesystem::path &path, FILE *stream) {
  FILE *file=nullptr;
  if(_wfopen_s(&file,path.c_str(),L"rb")!=0)return false;
  std::array<char,16384> bytes{};
  for(;;) {
    const auto count=std::fread(bytes.data(),1,bytes.size(),file);
    if(count && std::fwrite(bytes.data(),1,count,stream)!=count){std::fclose(file);return false;}
    if(count<bytes.size())break;
  }
  const auto ok=!std::ferror(file);std::fclose(file);std::fflush(stream);return ok;
}
}

int main() {
  SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
  std::array<wchar_t,32768> name{};
  const auto size=GetModuleFileNameW(nullptr,name.data(),static_cast<DWORD>(name.size()));
  if(size==0 || size==name.size())return 70;
  const auto directory=std::filesystem::path(name.data()).parent_path();
  const auto child=directory/L"hyle_cpu_best_of_sealed.exe";
  const auto stdout_path=directory/L"f4-child.stdout.tmp";
  const auto stderr_path=directory/L"f4-child.stderr.tmp";
  std::fprintf(stderr,"# CPU_BEST_OF_TRANSPORT pairs=%zu child=sealed-original-aa no_math=1\n",std::size(kP1Pairs));
  for(const auto &pair:kP1Pairs) {
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
    const auto output=CreateFileW(stdout_path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&attributes,CREATE_ALWAYS,FILE_ATTRIBUTE_TEMPORARY,nullptr);
    const auto error=CreateFileW(stderr_path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&attributes,CREATE_ALWAYS,FILE_ATTRIBUTE_TEMPORARY,nullptr);
    if(output==INVALID_HANDLE_VALUE || error==INVALID_HANDLE_VALUE) {
      if(output!=INVALID_HANDLE_VALUE)CloseHandle(output);
      if(error!=INVALID_HANDLE_VALUE)CloseHandle(error);
      return 71;
    }
    STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES|STARTF_USESHOWWINDOW;
    startup.wShowWindow=SW_HIDE;startup.hStdInput=GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput=output;startup.hStdError=error;
    PROCESS_INFORMATION process{};
    std::wstring command=L"\""+child.wstring()+L"\" --pair "+pair.case_id+L" "+pair.candidate+L" "+pair.peer;
    const auto created=CreateProcessW(child.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,directory.c_str(),&startup,&process);
    CloseHandle(output);CloseHandle(error);
    if(!created)return 72;
    const auto waited=WaitForSingleObject(process.hProcess,INFINITE);
    DWORD code=73;
    const auto read=GetExitCodeProcess(process.hProcess,&code);
    CloseHandle(process.hThread);CloseHandle(process.hProcess);
    if(!forward(stdout_path,stdout) || !forward(stderr_path,stderr))return 74;
    DeleteFileW(stdout_path.c_str());DeleteFileW(stderr_path.c_str());
    if(waited!=WAIT_OBJECT_0 || !read || code!=0)return static_cast<int>(code);
  }
  return 0;
}
