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
        if (args.length != 1) {
            System.err.println(
                    "usage: HyleCoreFfiSmoke "
                            + "<absolute-path-to-hyle_core-shared-library>");
            System.exit(2);
        }

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
            MethodHandle productVersionFunction = bind(
                    lookup,
                    "hyle_version",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT));
            MethodHandle productStringFunction = bind(
                    lookup,
                    "hyle_version_string",
                    FunctionDescriptor.of(ValueLayout.ADDRESS));

            int ffiAbi = (int) ffiAbiFunction.invokeExact();
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
            if (ffiAbi != EXPECTED_FFI_ABI
                    || productVersion != EXPECTED_PRODUCT_VERSION
                    || !EXPECTED_PRODUCT_STRING.equals(productString)) {
                throw new IllegalStateException(String.format(
                        "HYLE core ABI mismatch: ffi=0x%08x, "
                                + "product=0x%08x, string=%s; expected "
                                + "ffi=0x00010000, product=0x00000100, "
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
