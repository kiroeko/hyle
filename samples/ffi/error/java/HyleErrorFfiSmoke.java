import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public final class HyleErrorFfiSmoke {
    private static final int EXPECTED_FFI_ABI = 0x0001_0000;
    private static final int STATUS_OK = 0;
    private static final int STATUS_RESOURCE_EXHAUSTED = 8;
    private static final int STATUS_UNAVAILABLE = 14;
    private static final int NATIVE_NONE = 0;
    private static final int NATIVE_VULKAN = 3;
    private static final byte[] EXPECTED_MESSAGE = {
        'd', 'e', 'v', 0, 'l', 'o', 's', 't'
    };
    private static final Linker LINKER = Linker.nativeLinker();

    private HyleErrorFfiSmoke() {}

    private static MethodHandle bind(
            SymbolLookup lookup,
            String artifact,
            String symbol,
            FunctionDescriptor descriptor) {
        MemorySegment address = lookup.find(symbol).orElseThrow(
                () -> new UnsatisfiedLinkError(
                        "The exact " + artifact + " does not export " + symbol));
        return LINKER.downcallHandle(address, descriptor);
    }

    private static void requireCompatible(int actual, String artifact) {
        int requiredMajor = EXPECTED_FFI_ABI >>> 16;
        int requiredMinor = EXPECTED_FFI_ABI & 0xffff;
        int actualMajor = actual >>> 16;
        int actualMinor = actual & 0xffff;
        if (actualMajor != requiredMajor || actualMinor < requiredMinor) {
            throw new IllegalStateException(String.format(
                    "Incompatible %s FFI ABI: runtime=0x%08x, "
                            + "binding requires major=%d, minor>=%d.",
                    artifact, actual, requiredMajor, requiredMinor));
        }
    }

    private static Path exactLibrary(String argument, String artifact) {
        Path path = Path.of(argument).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException(
                    "The requested " + artifact + " does not exist: " + path);
        }
        return path;
    }

    private static byte[] copyBytes(MemorySegment buffer, int length) {
        byte[] result = new byte[length];
        for (int index = 0; index < length; ++index) {
            result[index] = buffer.get(ValueLayout.JAVA_BYTE, index);
        }
        return result;
    }

    private static final class OwnedStatus implements AutoCloseable {
        private final MethodHandle destroy;
        private MemorySegment handle;

        OwnedStatus(MemorySegment handle, MethodHandle destroy) {
            if (handle.address() == 0) {
                throw new IllegalArgumentException(
                        "an owned failure handle cannot be NULL");
            }
            this.handle = handle;
            this.destroy = destroy;
        }

        MemorySegment borrow() {
            if (handle.address() == 0) {
                throw new IllegalStateException("status handle already destroyed");
            }
            return handle;
        }

        @Override
        public void close() {
            MemorySegment owned = handle;
            if (owned.address() == 0) {
                return;
            }
            handle = MemorySegment.NULL;
            try {
                destroy.invokeExact(owned);
            } catch (Throwable error) {
                throw new IllegalStateException("hyle_status_destroy failed", error);
            }
        }
    }

    private static void runIncompatible(Path library) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            SymbolLookup lookup = SymbolLookup.libraryLookup(library, arena);
            MethodHandle bootstrap = bind(
                    lookup,
                    "incompatible hyle_error fixture",
                    "hyle_error_ffi_abi_version",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT));
            int actual = (int) bootstrap.invokeExact();
            int requiredMajor = EXPECTED_FFI_ABI >>> 16;
            int actualMajor = actual >>> 16;
            if (actualMajor == requiredMajor
                    && (actual & 0xffff) >= (EXPECTED_FFI_ABI & 0xffff)) {
                throw new IllegalStateException(String.format(
                        "Expected an incompatible hyle_error FFI ABI, but "
                                + "runtime 0x%08x is compatible.",
                        actual));
            }

            // The mismatch is established before any status symbol is resolved
            // or called. These post-rejection lookups only validate the fixture.
            String[] forbidden = {
                "hyle_status_destroy",
                "hyle_status_get_code",
                "hyle_status_get_native_error",
                "hyle_status_message",
                "hyle_status_to_string"
            };
            for (String symbol : forbidden) {
                if (lookup.find(symbol).isPresent()) {
                    throw new IllegalStateException(
                            "The incompatible fixture unexpectedly exports " + symbol);
                }
            }
            System.out.printf(
                    "Java correctly rejected incompatible hyle_error FFI ABI "
                            + "before status-symbol use: runtime=0x%08x, "
                            + "required=0x%08x; status symbols absent%n",
                    actual, EXPECTED_FFI_ABI);
        }
    }

    private static void runPositive(Path errorLibrary, Path producerLibrary)
            throws Throwable {
        try (Arena libraryArena = Arena.ofConfined();
                Arena dataArena = Arena.ofConfined()) {
            SymbolLookup errorLookup =
                    SymbolLookup.libraryLookup(errorLibrary, libraryArena);
            MethodHandle errorBootstrap = bind(
                    errorLookup,
                    "hyle_error library",
                    "hyle_error_ffi_abi_version",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT));
            int errorAbi = (int) errorBootstrap.invokeExact();
            requireCompatible(errorAbi, "hyle_error library");

            // Resolve the remaining hyle_error surface only after its own
            // exact artifact has passed bootstrap negotiation.
            MethodHandle destroy = bind(
                    errorLookup,
                    "hyle_error library",
                    "hyle_status_destroy",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            MethodHandle getCode = bind(
                    errorLookup,
                    "hyle_error library",
                    "hyle_status_get_code",
                    FunctionDescriptor.of(ValueLayout.JAVA_SHORT, ValueLayout.ADDRESS));
            MethodHandle getNative = bind(
                    errorLookup,
                    "hyle_error library",
                    "hyle_status_get_native_error",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS));
            MethodHandle message = bind(
                    errorLookup,
                    "hyle_error library",
                    "hyle_status_message",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG));
            MethodHandle toString = bind(
                    errorLookup,
                    "hyle_error library",
                    "hyle_status_to_string",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG));

            SymbolLookup producerLookup =
                    SymbolLookup.libraryLookup(producerLibrary, libraryArena);
            MethodHandle producerBootstrap = bind(
                    producerLookup,
                    "status producer fixture",
                    "hyle_error_status_fixture_ffi_abi_version",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT));
            int producerAbi = (int) producerBootstrap.invokeExact();
            requireCompatible(producerAbi, "status producer fixture");

            // The producer's deliberate-failure entry points are also resolved
            // only after that exact artifact has accepted the ABI generation.
            MethodHandle produceOk = bind(
                    producerLookup,
                    "status producer fixture",
                    "hyle_error_status_fixture_ok",
                    FunctionDescriptor.of(ValueLayout.ADDRESS));
            MethodHandle produceCodeOnly = bind(
                    producerLookup,
                    "status producer fixture",
                    "hyle_error_status_fixture_code_only",
                    FunctionDescriptor.of(ValueLayout.ADDRESS));
            MethodHandle produceMessageNative = bind(
                    producerLookup,
                    "status producer fixture",
                    "hyle_error_status_fixture_message_native",
                    FunctionDescriptor.of(ValueLayout.ADDRESS));

            MemorySegment ok = (MemorySegment) produceOk.invokeExact();
            if (ok.address() != 0) {
                throw new IllegalStateException("OK must cross the FFI as NULL");
            }
            short okCode = (short) getCode.invokeExact(MemorySegment.NULL);
            long okMessage = (long) message.invokeExact(
                    MemorySegment.NULL, MemorySegment.NULL, 0L);
            MemorySegment emptyDomain = dataArena.allocate(ValueLayout.JAVA_SHORT);
            MemorySegment emptyNativeCode = dataArena.allocate(ValueLayout.JAVA_LONG);
            emptyDomain.set(ValueLayout.JAVA_SHORT, 0, (short) 99);
            emptyNativeCode.set(ValueLayout.JAVA_LONG, 0, 99L);
            int okHasNative = (int) getNative.invokeExact(
                    MemorySegment.NULL, emptyDomain, emptyNativeCode);
            if ((okCode & 0xffff) != STATUS_OK || okMessage != 0
                    || okHasNative != 0
                    || (emptyDomain.get(ValueLayout.JAVA_SHORT, 0) & 0xffff)
                            != NATIVE_NONE
                    || emptyNativeCode.get(ValueLayout.JAVA_LONG, 0) != 0) {
                throw new IllegalStateException("NULL did not query as complete OK");
            }
            destroy.invokeExact(MemorySegment.NULL);

            MemorySegment codeFirst =
                    (MemorySegment) produceCodeOnly.invokeExact();
            MemorySegment codeSecond =
                    (MemorySegment) produceCodeOnly.invokeExact();
            try (OwnedStatus first = new OwnedStatus(codeFirst, destroy);
                    OwnedStatus second = new OwnedStatus(codeSecond, destroy)) {
                if (first.borrow().address() != second.borrow().address()) {
                    throw new IllegalStateException(
                            "code-only failures did not use the process token");
                }
                short code = (short) getCode.invokeExact(first.borrow());
                long length = (long) message.invokeExact(
                        first.borrow(), MemorySegment.NULL, 0L);
                MemorySegment domain = dataArena.allocate(ValueLayout.JAVA_SHORT);
                MemorySegment nativeCode = dataArena.allocate(ValueLayout.JAVA_LONG);
                int hasNative = (int) getNative.invokeExact(
                        first.borrow(), domain, nativeCode);
                if ((code & 0xffff) != STATUS_RESOURCE_EXHAUSTED
                        || length != 0 || hasNative != 0
                        || (domain.get(ValueLayout.JAVA_SHORT, 0) & 0xffff)
                                != NATIVE_NONE
                        || nativeCode.get(ValueLayout.JAVA_LONG, 0) != 0) {
                    throw new IllegalStateException(
                            "code-only status payload mismatch");
                }
            }

            MemorySegment richHandle =
                    (MemorySegment) produceMessageNative.invokeExact();
            try (OwnedStatus rich = new OwnedStatus(richHandle, destroy)) {
                short code = (short) getCode.invokeExact(rich.borrow());
                MemorySegment domain = dataArena.allocate(ValueLayout.JAVA_SHORT);
                MemorySegment nativeCode = dataArena.allocate(ValueLayout.JAVA_LONG);
                int hasNative = (int) getNative.invokeExact(
                        rich.borrow(), domain, nativeCode);
                if ((code & 0xffff) != STATUS_UNAVAILABLE || hasNative != 1
                        || (domain.get(ValueLayout.JAVA_SHORT, 0) & 0xffff)
                                != NATIVE_VULKAN
                        || nativeCode.get(ValueLayout.JAVA_LONG, 0) != -4) {
                    throw new IllegalStateException(
                            "message/native status metadata mismatch");
                }

                long required = (long) message.invokeExact(
                        rich.borrow(), MemorySegment.NULL, 0L);
                if (required != EXPECTED_MESSAGE.length) {
                    throw new IllegalStateException(
                            "embedded-NUL message length mismatch: " + required);
                }
                MemorySegment copied = dataArena.allocate(required + 1, 1);
                long copiedRequired = (long) message.invokeExact(
                        rich.borrow(), copied, required + 1);
                if (copiedRequired != required
                        || copied.get(ValueLayout.JAVA_BYTE, required) != 0
                        || !Arrays.equals(
                                copyBytes(copied, Math.toIntExact(required)),
                                EXPECTED_MESSAGE)) {
                    throw new IllegalStateException(
                            "embedded-NUL message bytes were not preserved");
                }

                long textRequired = (long) toString.invokeExact(
                        rich.borrow(), MemorySegment.NULL, 0L);
                MemorySegment tiny = dataArena.allocate(5, 1);
                long tinyRequired = (long) toString.invokeExact(
                        rich.borrow(), tiny, 5L);
                if (textRequired <= required || tinyRequired != textRequired
                        || tiny.get(ValueLayout.JAVA_BYTE, 4) != 0) {
                    throw new IllegalStateException(
                            "hyle_status_to_string caller-buffer contract failed");
                }
            }

            System.out.printf(
                    "Java direct HYLE error FFI smoke passed: error=0x%08x, "
                            + "producer=0x%08x, NULL/code-only/embedded-NUL "
                            + "message+native handles destroyed%n",
                    errorAbi, producerAbi);
        }
    }

    public static void main(String[] args) throws Throwable {
        if (args.length != 2) {
            System.err.println(
                    "usage: HyleErrorFfiSmoke <exact-hyle_error-library> "
                            + "<exact-status-producer-library>\n"
                            + "   or: HyleErrorFfiSmoke "
                            + "<exact-incompatible-library> --expect-incompatible");
            System.exit(2);
        }

        if ("--expect-incompatible".equals(args[1])) {
            runIncompatible(exactLibrary(args[0], "incompatible fixture"));
            return;
        }
        runPositive(
                exactLibrary(args[0], "hyle_error shared library"),
                exactLibrary(args[1], "status producer fixture"));
    }
}
