@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 && cmake --build "C:\public\project-hyle\hyle\.scratch\build\m2-vulkan-indexed" --target hyle_compute_route_vulkan_tests hyle_graph_route_vulkan_tests -j 2
