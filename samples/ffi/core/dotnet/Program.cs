using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

internal static partial class HyleCoreNative
{
    // This is deliberately not an operating-system library name. The resolver
    // below is the only supported route to the exact file supplied by the
    // smoke runner.
    private const string ImportName = "hyle_core_exact_path";
    private static readonly object LoadGate = new();
    private static string? exactLibraryPath;
    private static IntPtr libraryHandle;

    [LibraryImport(ImportName, EntryPoint = "hyle_ffi_abi_version")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial uint FfiAbiVersion();

    [LibraryImport(ImportName, EntryPoint = "hyle_version")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial uint ProductVersion();

    [LibraryImport(ImportName, EntryPoint = "hyle_version_string")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial IntPtr ProductVersionString();

    internal static void BindExactLibrary(string path)
    {
        string canonicalPath = Path.GetFullPath(path);
        if (!File.Exists(canonicalPath))
        {
            throw new FileNotFoundException(
                "The requested hyle_core shared library does not exist.",
                canonicalPath);
        }

        lock (LoadGate)
        {
            if (exactLibraryPath is not null)
            {
                throw new InvalidOperationException(
                    "The hyle_core resolver has already been configured.");
            }
            exactLibraryPath = canonicalPath;
            NativeLibrary.SetDllImportResolver(
                typeof(HyleCoreNative).Assembly, ResolveExactLibrary);
        }
    }

    internal static bool HasExport(string symbol)
    {
        lock (LoadGate)
        {
            if (libraryHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    "The exact hyle_core library has not been loaded.");
            }
            return NativeLibrary.TryGetExport(libraryHandle, symbol, out _);
        }
    }

    private static IntPtr ResolveExactLibrary(
        string libraryName,
        Assembly assembly,
        DllImportSearchPath? searchPath)
    {
        _ = assembly;
        _ = searchPath;
        if (!string.Equals(libraryName, ImportName, StringComparison.Ordinal))
        {
            return IntPtr.Zero;
        }

        lock (LoadGate)
        {
            if (exactLibraryPath is null)
            {
                throw new DllNotFoundException(
                    "The exact hyle_core library path was not configured.");
            }
            if (libraryHandle == IntPtr.Zero)
            {
                libraryHandle = NativeLibrary.Load(exactLibraryPath);
            }
            return libraryHandle;
        }
    }
}

internal static class Program
{
    private const uint ExpectedFfiAbi = 0x0001_0000U;
    private const uint ExpectedProductVersion = 0x0000_0100U;
    private const string ExpectedProductString = "0.1.0";

    private static int Main(string[] args)
    {
        if (args.Length < 1 || args.Length > 2 ||
            (args.Length == 2 &&
             !string.Equals(
                 args[1], "--expect-incompatible", StringComparison.Ordinal)))
        {
            Console.Error.WriteLine(
                "usage: HyleCoreFfiSmoke <absolute-path-to-shared-library> " +
                "[--expect-incompatible]");
            return 2;
        }
        bool expectIncompatible = args.Length == 2;

        try
        {
            HyleCoreNative.BindExactLibrary(args[0]);
            uint ffiAbi = HyleCoreNative.FfiAbiVersion();
            uint requiredMajor = ExpectedFfiAbi >> 16;
            uint requiredMinor = ExpectedFfiAbi & 0xffffU;
            uint runtimeMajor = ffiAbi >> 16;
            uint runtimeMinor = ffiAbi & 0xffffU;
            bool compatible = runtimeMajor == requiredMajor &&
                              runtimeMinor >= requiredMinor;
            if (!compatible)
            {
                if (expectIncompatible)
                {
                    if (HyleCoreNative.HasExport("hyle_version") ||
                        HyleCoreNative.HasExport("hyle_version_string"))
                    {
                        throw new InvalidOperationException(
                            "The incompatible fixture must export only the ABI bootstrap.");
                    }
                    Console.WriteLine(
                        ".NET correctly rejected incompatible HYLE FFI ABI: " +
                        $"runtime=0x{ffiAbi:x8}, required=0x{ExpectedFfiAbi:x8}; " +
                        "product symbols absent");
                    return 0;
                }
                throw new InvalidOperationException(
                    $"Incompatible HYLE FFI ABI: runtime=0x{ffiAbi:x8}, " +
                    $"binding requires major={requiredMajor}, minor>={requiredMinor}.");
            }
            if (expectIncompatible)
            {
                throw new InvalidOperationException(
                    "Expected an incompatible HYLE FFI ABI, but runtime " +
                    $"0x{ffiAbi:x8} is compatible.");
            }

            // LibraryImport resolves these entry points lazily on first call;
            // do not touch them until the global ABI bootstrap has succeeded.
            uint productVersion = HyleCoreNative.ProductVersion();
            IntPtr productStringPointer = HyleCoreNative.ProductVersionString();
            if (productStringPointer == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    "hyle_version_string returned a null borrowed pointer.");
            }

            string? productString = Marshal.PtrToStringUTF8(productStringPointer);
            if (productVersion != ExpectedProductVersion ||
                !string.Equals(
                    productString, ExpectedProductString, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"HYLE core product mismatch: ffi=0x{ffiAbi:x8}, " +
                    $"product=0x{productVersion:x8}, " +
                    $"string={productString ?? "<null>"}; expected " +
                    "product=0x00000100, string=0.1.0.");
            }

            Console.WriteLine(
                $".NET direct HYLE core FFI smoke passed: " +
                $"ffi=0x{ffiAbi:x8}, product=0x{productVersion:x8}, " +
                $"string={productString}");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($".NET HYLE core FFI smoke failed: {error}");
            return 1;
        }
    }
}
