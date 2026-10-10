@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 > "C:\h7\h2-apple-acceptance\platform-off\env.log" 2>&1
if errorlevel 1 exit /b 1
cl /nologo /TP /EHsc /std:c++latest /MD /permissive- /Zc:__cplusplus /utf-8 /Zc:preprocessor /Zs /I"C:\public\project-hyle\hyle\include" "C:\h7\h2-apple-acceptance\platform-off\0.cc" > "C:\h7\h2-apple-acceptance\platform-off\0.log" 2>&1
if errorlevel 1 exit /b 1
cl /nologo /TP /EHsc /std:c++latest /MD /permissive- /Zc:__cplusplus /utf-8 /Zc:preprocessor /Zs /I"C:\public\project-hyle\hyle\include" "C:\h7\h2-apple-acceptance\platform-off\1.cc" > "C:\h7\h2-apple-acceptance\platform-off\1.log" 2>&1
if errorlevel 1 exit /b 1
cl /nologo /TP /EHsc /std:c++latest /MD /permissive- /Zc:__cplusplus /utf-8 /Zc:preprocessor /Zs /I"C:\public\project-hyle\hyle\include" "C:\h7\h2-apple-acceptance\platform-off\2.cc" > "C:\h7\h2-apple-acceptance\platform-off\2.log" 2>&1
if errorlevel 1 exit /b 1
cl /nologo /TP /EHsc /std:c++latest /MD /permissive- /Zc:__cplusplus /utf-8 /Zc:preprocessor /Zs /I"C:\public\project-hyle\hyle\include" "C:\h7\h2-apple-acceptance\platform-off\3.cc" > "C:\h7\h2-apple-acceptance\platform-off\3.log" 2>&1
if errorlevel 1 exit /b 1
