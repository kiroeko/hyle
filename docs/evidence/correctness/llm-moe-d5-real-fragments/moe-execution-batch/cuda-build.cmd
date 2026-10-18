@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 && cmake --build "C:\public\project-hyle\hyle\.scratch\b-cuda-d5" --target hyle_compute_route_cuda_tests hyle_graph_route_cuda_tests -j 2
