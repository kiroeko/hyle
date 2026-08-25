import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

public final class HyleCoreFfiSmoke {
    private static final int EXPECTED_FFI_ABI = 0x0001_0000;
    private static final int EXPECTED_PRODUCT_VERSION = 0x0000_0100;
    private static final String EXPECTED_PRODUCT_STRING = "0.1.0";
    private static final Linker LINKER = Linker.nativeLinker();

    private HyleCoreFfiSmoke() {}

    private static MethodHandle bind(
            SymbolLookup lookup, String symbol, FunctionDescriptor descriptor) {
        MemorySegment address = lookup.find(symbol).orElseThrow(
                () -> new UnsatisfiedLinkError(
                        "The exact hyle_core library does not export " + symbol));
        return LINKER.downcallHandle(address, descriptor);
    }

    public static void main(String[] args) throws Throwable {
        if (args.length < 1 || args.length > 2
                || (args.length == 2 && !"--expect-incompatible".equals(args[1]))) {
            System.err.println(
                    "usage: HyleCoreFfiSmoke "
                            + "<absolute-path-to-shared-library> "
                            + "[--expect-incompatible]");
            System.exit(2);
        }
        boolean expectIncompatible = args.length == 2;

        Path library = Path.of(args[0]).toAbsolutePath().normalize();
        if (!Files.isRegularFile(library)) {
            throw new IllegalArgumentException(
                    "The requested hyle_core shared library does not exist: "
                            + library);
        }

        try (Arena arena = Arena.ofConfined()) {
            SymbolLookup lookup = SymbolLookup.libraryLookup(library, arena);
            MethodHandle ffiAbiFunction = bind(
                    lookup,
                    "hyle_ffi_abi_version",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT));

            int ffiAbi = (int) ffiAbiFunction.invokeExact();
            int requiredMajor = EXPECTED_FFI_ABI >>> 16;
            int requiredMinor = EXPECTED_FFI_ABI & 0xffff;
            int runtimeMajor = ffiAbi >>> 16;
            int runtimeMinor = ffiAbi & 0xffff;
            boolean compatible = runtimeMajor == requiredMajor
                    && runtimeMinor >= requiredMinor;
            if (!compatible) {
                if (expectIncompatible) {
                    System.out.printf(
                            "Java correctly rejected incompatible HYLE FFI ABI: "
                                    + "runtime=0x%08x, required=0x%08x%n",
                            ffiAbi, EXPECTED_FFI_ABI);
                    return;
                }
                throw new IllegalStateException(String.format(
                        "Incompatible HYLE FFI ABI: runtime=0x%08x, "
                                + "binding requires major=%d, minor>=%d.",
                        ffiAbi, requiredMajor, requiredMinor));
            }
            if (expectIncompatible) {
                throw new IllegalStateException(String.format(
                        "Expected an incompatible HYLE FFI ABI, but runtime "
                                + "0x%08x is compatible.",
                        ffiAbi));
            }

            // Resolve the remaining required symbols only after the global ABI
            // bootstrap has accepted this exact native artifact.
            MethodHandle productVersionFunction = bind(
                    lookup,
                    "hyle_version",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT));
            MethodHandle productStringFunction = bind(
                    lookup,
                    "hyle_version_string",
                    FunctionDescriptor.of(ValueLayout.ADDRESS));

            int productVersion = (int) productVersionFunction.invokeExact();
            MemorySegment productStringPointer =
                    (MemorySegment) productStringFunction.invokeExact();
            if (productStringPointer.address() == 0) {
                throw new IllegalStateException(
                        "hyle_version_string returned a null borrowed pointer");
            }

            // A native address return has zero spatial bounds. Reinterpret only
            // enough borrowed storage to read this tiny NUL-terminated version.
            String productString = productStringPointer.reinterpret(64).getString(0);
            if (productVersion != EXPECTED_PRODUCT_VERSION
                    || !EXPECTED_PRODUCT_STRING.equals(productString)) {
                throw new IllegalStateException(String.format(
                        "HYLE core product mismatch: ffi=0x%08x, "
                                + "product=0x%08x, string=%s; expected "
                                + "product=0x00000100, "
                                + "string=0.1.0.",
                        ffiAbi, productVersion, productString));
            }

            System.out.printf(
                    "Java direct HYLE core FFI smoke passed: "
                            + "ffi=0x%08x, product=0x%08x, string=%s%n",
                    ffiAbi, productVersion, productString);
        }
    }
}
