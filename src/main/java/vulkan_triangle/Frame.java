package vulkan_triangle;

import java.nio.LongBuffer;

import static org.lwjgl.system.MemoryStack.stackGet;

/**
 * Wrapper for the sync objects of an in-flight frame.
 * The sync objects must be deleted manually!
 */
public record Frame(long imageAvailableSemaphore, long renderFinishedSemaphore, long fence) {
    public LongBuffer pImageAvailableSemaphore() {
        return stackGet().longs(imageAvailableSemaphore);
    }

    public LongBuffer pRenderFinishedSemaphore() {
        return stackGet().longs(renderFinishedSemaphore);
    }

    public LongBuffer pFence() {
        return stackGet().longs(fence);
    }
}
