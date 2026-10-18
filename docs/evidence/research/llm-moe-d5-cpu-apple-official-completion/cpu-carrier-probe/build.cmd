@echo off
call "C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
set VSLANG=1033
cl /nologo /std:c++20 /O2 /fp:strict /EHsc /MD /I"C:\Program Files\Python312\Lib\site-packages\torch\include" cxx.cc /Fe:cxx.exe /link /LIBPATH:"C:\public\project-hyle\hyle\.scratch\validation\d5-cpu-apple-official-sources\cpu-sdk-subset\lib" torch_cpu.lib c10.lib
if errorlevel 1 exit /b %errorlevel%
dumpbin /dependents torch_cpu.dll
dumpbin /dependents cxx.exe
exit /b %errorlevel%
