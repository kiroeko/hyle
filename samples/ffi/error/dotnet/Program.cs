using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

internal static partial class HyleErrorNative
{
    // Neither name is an OS library name. The resolver below maps each import
    // to the exact artifact path supplied by the smoke runner.
    private const string ErrorImportName = "hyle_error_exact_path";
    private const string ProducerImportName = "hyle_error_producer_exact_path";
    private static readonly object LoadGate = new();
    private static string? errorLibraryPath;
    private static string? producerLibraryPath;
    private static IntPtr errorLibraryHandle;
    private static IntPtr producerLibraryHandle;

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_error_ffi_abi_version")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial uint ErrorFfiAbiVersion();

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_destroy")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial void StatusDestroy(IntPtr status);

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_get_code")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial ushort StatusGetCodeRaw(IntPtr status);

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_get_code")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial ushort StatusGetCode(OwnedStatus status);

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_get_native_error")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial int StatusGetNativeErrorRaw(
        IntPtr status, out ushort domain, out long nativeCode);

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_get_native_error")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial int StatusGetNativeError(
        OwnedStatus status, out ushort domain, out long nativeCode);

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_message")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static unsafe partial ulong StatusMessageRaw(
        IntPtr status, byte* buffer, ulong capacity);

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_message")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static unsafe partial ulong StatusMessage(
        OwnedStatus status, byte* buffer, ulong capacity);

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_to_string")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static unsafe partial ulong StatusToStringRaw(
        IntPtr status, byte* buffer, ulong capacity);

    [LibraryImport(ErrorImportName, EntryPoint = "hyle_status_to_string")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static unsafe partial ulong StatusToString(
        OwnedStatus status, byte* buffer, ulong capacity);

    [LibraryImport(
        ProducerImportName,
        EntryPoint = "hyle_error_status_fixture_ffi_abi_version")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial uint ProducerFfiAbiVersion();

    [LibraryImport(
        ProducerImportName,
        EntryPoint = "hyle_error_status_fixture_ok")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial IntPtr ProduceOk();

    [LibraryImport(
        ProducerImportName,
        EntryPoint = "hyle_error_status_fixture_code_only")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial IntPtr ProduceCodeOnly();

    [LibraryImport(
        ProducerImportName,
        EntryPoint = "hyle_error_status_fixture_message_native")]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    internal static partial IntPtr ProduceMessageNative();

    internal static void BindExactLibraries(string errorPath, string? producerPath)
    {
        string canonicalErrorPath = RequireFile(errorPath, "hyle_error library");
        string? canonicalProducerPath = producerPath is null
            ? null
            : RequireFile(producerPath, "status producer fixture");

        lock (LoadGate)
        {
            if (errorLibraryPath is not null)
            {
                throw new InvalidOperationException(
                    "The exact-library resolver has already been configured.");
            }
            errorLibraryPath = canonicalErrorPath;
            producerLibraryPath = canonicalProducerPath;
            NativeLibrary.SetDllImportResolver(
                typeof(HyleErrorNative).Assembly, ResolveExactLibrary);
        }
    }

    internal static bool HasErrorExport(string symbol)
    {
        lock (LoadGate)
        {
            if (errorLibraryHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    "The exact hyle_error artifact has not been loaded.");
            }
            return NativeLibrary.TryGetExport(errorLibraryHandle, symbol, out _);
        }
    }

    private static string RequireFile(string path, string artifact)
    {
        string canonicalPath = Path.GetFullPath(path);
        if (!File.Exists(canonicalPath))
        {
            throw new FileNotFoundException(
                $"The requested {artifact} does not exist.", canonicalPath);
        }
        return canonicalPath;
    }

    private static IntPtr ResolveExactLibrary(
        string libraryName,
        Assembly assembly,
        DllImportSearchPath? searchPath)
    {
        _ = assembly;
        _ = searchPath;
        lock (LoadGate)
        {
            if (string.Equals(
                    libraryName, ErrorImportName, StringComparison.Ordinal))
            {
                if (errorLibraryPath is null)
                {
                    throw new DllNotFoundException(
                        "The exact hyle_error path was not configured.");
                }
                if (errorLibraryHandle == IntPtr.Zero)
                {
                    errorLibraryHandle = NativeLibrary.Load(errorLibraryPath);
                }
                return errorLibraryHandle;
            }
            if (string.Equals(
                    libraryName, ProducerImportName, StringComparison.Ordinal))
            {
                if (producerLibraryPath is null)
                {
                    throw new DllNotFoundException(
                        "The exact status-producer path was not configured.");
                }
                if (producerLibraryHandle == IntPtr.Zero)
                {
                    producerLibraryHandle = NativeLibrary.Load(producerLibraryPath);
                }
                return producerLibraryHandle;
            }
            return IntPtr.Zero;
        }
    }
}

internal sealed class OwnedStatus : SafeHandle
{
    internal OwnedStatus(IntPtr owned) : base(IntPtr.Zero, ownsHandle: true)
    {
        if (owned == IntPtr.Zero)
        {
            throw new ArgumentException(
                "An owned failure handle cannot be NULL.", nameof(owned));
        }
        SetHandle(owned);
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    internal IntPtr IdentityForTestOnly()
    {
        if (IsClosed || IsInvalid)
        {
            throw new ObjectDisposedException(nameof(OwnedStatus));
        }
        return DangerousGetHandle();
    }

    protected override bool ReleaseHandle()
    {
        HyleErrorNative.StatusDestroy(handle);
        return true;
    }
}

internal static class Program
{
    private const uint ExpectedFfiAbi = 0x0001_0000U;
    private const ushort StatusOk = 0;
    private const ushort StatusResourceExhausted = 8;
    private const ushort StatusUnavailable = 14;
    private const ushort NativeNone = 0;
    private const ushort NativeVulkan = 3;
    private static readonly byte[] ExpectedMessage =
        { (byte)'d', (byte)'e', (byte)'v', 0,
          (byte)'l', (byte)'o', (byte)'s', (byte)'t' };

    private static void RequireCompatible(uint actual, string artifact)
    {
        uint requiredMajor = ExpectedFfiAbi >> 16;
        uint requiredMinor = ExpectedFfiAbi & 0xffffU;
        uint actualMajor = actual >> 16;
        uint actualMinor = actual & 0xffffU;
        if (actualMajor != requiredMajor || actualMinor < requiredMinor)
        {
            throw new InvalidOperationException(
                $"Incompatible {artifact} FFI ABI: runtime=0x{actual:x8}, " +
                $"binding requires major={requiredMajor}, minor>={requiredMinor}.");
        }
    }

    private static int RunIncompatible(string library)
    {
        HyleErrorNative.BindExactLibraries(library, producerPath: null);
        uint actual = HyleErrorNative.ErrorFfiAbiVersion();
        uint requiredMajor = ExpectedFfiAbi >> 16;
        bool compatible = (actual >> 16) == requiredMajor &&
                          (actual & 0xffffU) >= (ExpectedFfiAbi & 0xffffU);
        if (compatible)
        {
            throw new InvalidOperationException(
                "Expected an incompatible hyle_error FFI ABI, but runtime " +
                $"0x{actual:x8} is compatible.");
        }

        // The mismatch is established before any status import is called.
        // These post-rejection lookups only validate the fixture's surface.
        string[] forbidden =
        {
            "hyle_status_destroy",
            "hyle_status_get_code",
            "hyle_status_get_native_error",
            "hyle_status_message",
            "hyle_status_to_string"
        };
        foreach (string symbol in forbidden)
        {
            if (HyleErrorNative.HasErrorExport(symbol))
            {
                throw new InvalidOperationException(
                    $"The incompatible fixture unexpectedly exports {symbol}.");
            }
        }
        Console.WriteLine(
            ".NET correctly rejected incompatible hyle_error FFI ABI before " +
            $"status-symbol use: runtime=0x{actual:x8}, " +
            $"required=0x{ExpectedFfiAbi:x8}; status symbols absent");
        return 0;
    }

    private static unsafe int RunPositive(
        string errorLibrary, string producerLibrary)
    {
        HyleErrorNative.BindExactLibraries(errorLibrary, producerLibrary);

        uint errorAbi = HyleErrorNative.ErrorFfiAbiVersion();
        RequireCompatible(errorAbi, "hyle_error library");

        // LibraryImport resolves entry points lazily. Do not touch any producer
        // source until that exact artifact's bootstrap has also passed.
        uint producerAbi = HyleErrorNative.ProducerFfiAbiVersion();
        RequireCompatible(producerAbi, "status producer fixture");

        IntPtr ok = HyleErrorNative.ProduceOk();
        ushort emptyDomain = 99;
        long emptyNativeCode = 99;
            int okHasNative = HyleErrorNative.StatusGetNativeErrorRaw(
                IntPtr.Zero, out emptyDomain, out emptyNativeCode);
            if (ok != IntPtr.Zero ||
                HyleErrorNative.StatusGetCodeRaw(IntPtr.Zero) != StatusOk ||
                HyleErrorNative.StatusMessageRaw(IntPtr.Zero, null, 0) != 0 ||
                okHasNative != 0 || emptyDomain != NativeNone || emptyNativeCode != 0)
        {
            throw new InvalidOperationException(
                "NULL did not cross or query as complete OK.");
        }
        HyleErrorNative.StatusDestroy(ok);

        IntPtr codeFirst = HyleErrorNative.ProduceCodeOnly();
        IntPtr codeSecond = HyleErrorNative.ProduceCodeOnly();
        using (OwnedStatus first = new(codeFirst))
        using (OwnedStatus second = new(codeSecond))
        {
            if (first.IdentityForTestOnly() != second.IdentityForTestOnly())
            {
                throw new InvalidOperationException(
                    "Code-only failures did not use the process token.");
            }
            int hasNative = HyleErrorNative.StatusGetNativeError(
                first, out ushort domain, out long nativeCode);
            if (HyleErrorNative.StatusGetCode(first) !=
                    StatusResourceExhausted ||
                HyleErrorNative.StatusMessage(first, null, 0) != 0 ||
                hasNative != 0 || domain != NativeNone || nativeCode != 0)
            {
                throw new InvalidOperationException(
                    "Code-only status payload mismatch.");
            }
        }

        IntPtr richPointer = HyleErrorNative.ProduceMessageNative();
        using (OwnedStatus rich = new(richPointer))
        {
            int hasNative = HyleErrorNative.StatusGetNativeError(
                rich, out ushort domain, out long nativeCode);
            if (HyleErrorNative.StatusGetCode(rich) != StatusUnavailable ||
                hasNative != 1 || domain != NativeVulkan || nativeCode != -4)
            {
                throw new InvalidOperationException(
                    "Message/native status metadata mismatch.");
            }

            ulong required = HyleErrorNative.StatusMessage(
                rich, null, 0);
            if (required != (ulong)ExpectedMessage.Length)
            {
                throw new InvalidOperationException(
                    $"Embedded-NUL message length mismatch: {required}.");
            }
            byte[] copied = new byte[checked((int)required + 1)];
            ulong copiedRequired;
            fixed (byte* buffer = copied)
            {
                copiedRequired = HyleErrorNative.StatusMessage(
                    rich, buffer, (ulong)copied.Length);
            }
            if (copiedRequired != required || copied[^1] != 0 ||
                !copied.AsSpan(0, ExpectedMessage.Length)
                    .SequenceEqual(ExpectedMessage))
            {
                throw new InvalidOperationException(
                    "Embedded-NUL message bytes were not preserved.");
            }

            ulong textRequired = HyleErrorNative.StatusToString(
                rich, null, 0);
            byte* tiny = stackalloc byte[5];
            ulong tinyRequired = HyleErrorNative.StatusToString(
                rich, tiny, 5);
            if (textRequired <= required || tinyRequired != textRequired ||
                tiny[4] != 0)
            {
                throw new InvalidOperationException(
                    "hyle_status_to_string caller-buffer contract failed.");
            }
        }

        Console.WriteLine(
            ".NET direct HYLE error FFI smoke passed: " +
            $"error=0x{errorAbi:x8}, producer=0x{producerAbi:x8}, " +
            "NULL/code-only/embedded-NUL message+native handles destroyed");
        return 0;
    }

    private static int Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine(
                "usage: HyleErrorFfiSmoke <exact-hyle_error-library> " +
                "<exact-status-producer-library>\n" +
                "   or: HyleErrorFfiSmoke <exact-incompatible-library> " +
                "--expect-incompatible");
            return 2;
        }

        try
        {
            return string.Equals(
                    args[1], "--expect-incompatible", StringComparison.Ordinal)
                ? RunIncompatible(args[0])
                : RunPositive(args[0], args[1]);
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($".NET HYLE error FFI smoke failed: {error}");
            return 1;
        }
    }
}
