"""One read-only Linux loader/device inventory; never a route correctness verdict."""
import ctypes as c
import ctypes.util
import json
from pathlib import Path
import platform
import shutil

assert platform.system() == 'Linux'
record = {'classification': 'environment inventory only; no HYLE route or performance qualification',
    'kernel': platform.release(), 'architecture': platform.machine(),
    'vulkaninfo': shutil.which('vulkaninfo'), 'glslangValidator': shutil.which('glslangValidator'),
    'devices': [], 'icds': {str(p): p.read_text() for p in Path('/usr/share/vulkan/icd.d').glob('*.json')}}

class Application(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('pApplicationName', c.c_char_p),
        ('applicationVersion', c.c_uint32), ('pEngineName', c.c_char_p),
        ('engineVersion', c.c_uint32), ('apiVersion', c.c_uint32)]

class Create(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('flags', c.c_uint32),
        ('pApplicationInfo', c.POINTER(Application)), ('enabledLayerCount', c.c_uint32),
        ('ppEnabledLayerNames', c.c_void_p), ('enabledExtensionCount', c.c_uint32),
        ('ppEnabledExtensionNames', c.c_void_p)]

def version(value):
    return f'{value >> 22}.{(value >> 12) & 1023}.{value & 4095}'

instance = c.c_void_p()
try:
    loader_name = ctypes.util.find_library('vulkan')
    record['loader'] = loader_name
    loader = c.CDLL(loader_name or 'libvulkan.so.1')
    loader.vkCreateInstance.argtypes = [c.POINTER(Create), c.c_void_p, c.POINTER(c.c_void_p)]
    loader.vkCreateInstance.restype = c.c_int32
    loader.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    loader.vkDestroyInstance.restype = None
    loader.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
    loader.vkEnumeratePhysicalDevices.restype = c.c_int32
    loader.vkGetPhysicalDeviceProperties.argtypes = [c.c_void_p, c.c_void_p]
    loader.vkGetPhysicalDeviceProperties.restype = None
    app = Application(sType=0, pApplicationName=b'hyle-r6-environment-inventory', apiVersion=(1 << 22) | (2 << 12))
    info = Create(sType=1, pApplicationInfo=c.pointer(app))
    result = loader.vkCreateInstance(c.byref(info), None, c.byref(instance))
    record['create_instance_result'] = result
    if result == 0:
        count = c.c_uint32()
        result = loader.vkEnumeratePhysicalDevices(instance, c.byref(count), None)
        record['enumerate_count_result'] = result
        if result == 0 and count.value:
            devices = (c.c_void_p * count.value)()
            result = loader.vkEnumeratePhysicalDevices(instance, c.byref(count), devices)
            record['enumerate_devices_result'] = result
            if result == 0:
                for device in devices[:count.value]:
                    # Oversized, naturally aligned backing for the core Vulkan properties structure.
                    properties = (c.c_uint64 * 512)()
                    loader.vkGetPhysicalDeviceProperties(device, c.byref(properties))
                    data = bytes(properties)
                    words = [int.from_bytes(data[i:i+4], 'little') for i in range(0, 20, 4)]
                    record['devices'].append({'api': version(words[0]), 'driver_version_raw': words[1],
                        'vendor_id': hex(words[2]), 'device_id': hex(words[3]), 'device_type': words[4],
                        'software_cpu_device': words[4] == 4,
                        'name': data[20:276].split(b'\0', 1)[0].decode(errors='replace')})
except Exception as error:
    record['error'] = str(error)
finally:
    if instance:
        loader.vkDestroyInstance(instance, None)
    print(json.dumps(record, indent=2))
