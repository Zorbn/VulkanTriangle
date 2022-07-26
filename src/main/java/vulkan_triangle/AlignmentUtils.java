package vulkan_triangle;

public final class AlignmentUtils {
    private AlignmentUtils() {}

    public static long align(long offset, int alignment) {
        return offset % alignment == 0 ? offset : ((offset - 1) | (alignment - 1)) + 1;
    }

    public static int align(int offset, int alignment) {
        return offset % alignment == 0 ? offset : ((offset - 1) | (alignment - 1)) + 1;
    }
}