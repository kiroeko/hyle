import ctypes as c
import json
import platform

class App(c.Structure):
    _fields_ = [('sType', c.c_int), ('pNext', c.c_void_p), ('pApplicationName', c.c_char_p),
                ('applicationVersion', c.c_uint32), ('pEngineName', c.c_char_p),
                ('engineVersion', c.c_uint32), ('apiVersion', c.c_uint32)]

class Create(c.Structure):
    _fields_ = [('sType', c.c_int), ('pNext', c.c_void_p), ('flags', c.c_uint32),
                ('pApplicationInfo', c.POINTER(App)), ('enabledLayerCount', c.c_uint32),
                ('ppEnabledLayerNames', c.c_void_p), ('enabledExtensionCount', c.c_uint32),
                ('ppEnabledExtensionNames', c.c_void_p)]

vk = c.CDLL('libvulkan.so.1')
vk.vkCreateInstance.argtypes = [c.POINTER(Create), c.c_void_p, c.POINTER(c.c_void_p)]
vk.vkCreateInstance.restype = c.c_int
vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.c_void_p]
vk.vkEnumeratePhysicalDevices.restype = c.c_int
vk.vkGetPhysicalDeviceProperties.argtypes = [c.c_void_p, c.c_void_p]
vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
app = App(sType=0, pApplicationName=b'HYLE F1 actual device preflight', apiVersion=(1<<22))
info = Create(sType=1, pApplicationInfo=c.pointer(app))
instance = c.c_void_p()
code = vk.vkCreateInstance(c.byref(info), None, c.byref(instance))
result = dict(os=platform.platform(), create_instance=code, devices=[])
if code == 0:
    try:
        count = c.c_uint32()
        code = vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None)
        result['enumerate'] = code
        if code == 0:
            devices = (c.c_void_p * count.value)()
            assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices) == 0
            for device in devices[:count.value]:
                # Fixed Vulkan 1.0 prefix; generous aligned storage for the full properties.
                properties = (c.c_uint64 * 512)()
                vk.vkGetPhysicalDeviceProperties(device, c.byref(properties))
                prefix = (c.c_uint32 * 5).from_buffer(properties)
                name = c.string_at(c.addressof(properties)+20, 256).split(b'\0')[0].decode()
                result['devices'].append(dict(name=name, api_version=prefix[0], driver_version=prefix[1],
                                              vendor_id=prefix[2], device_id=prefix[3], device_type=prefix[4]))
    finally:
        vk.vkDestroyInstance(instance, None)
result['qualified_gpu_present'] = any(device['device_type'] in (1, 2) for device in result['devices'])
print(json.dumps(result, indent=2))
