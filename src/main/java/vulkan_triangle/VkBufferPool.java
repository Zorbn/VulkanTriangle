package vulkan_triangle;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.NativeResource;
import org.lwjgl.vulkan.*;

import java.nio.LongBuffer;
import java.util.HashMap;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;

public class VkBufferPool implements NativeResource {
    public record Buffer(long handle, int memOffset, long size) { }

    // Alignment required by Vulkan spec.
    private static final int OFFSET_ALIGNMENT = 0x100;

    private final VkDevice device;
    private final long memoryHandle;
    private final HashMap<Long, Buffer> buffers = new HashMap<>();
    private int totalOffset;

    public VkBufferPool(VkPhysicalDevice physicalDevice, VkDevice device, long size, int usages, int properties) {
        this.device = device;

        try (MemoryStack stack = stackPush()) {
            LongBuffer pBufferMemory = stack.mallocLong(1);

            LongBuffer pTempBuffer = stack.mallocLong(1);
            VkBufferCreateInfo tempBufferInfo = VkBufferCreateInfo.calloc(stack);
            tempBufferInfo.sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
            tempBufferInfo.size(0x100);
            tempBufferInfo.usage(usages);

            if (vkCreateBuffer(device, tempBufferInfo, null, pTempBuffer) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create temp buffer during memory pool initialization!");
            }

            long tempBuffer = pTempBuffer.get(0);

            VkMemoryRequirements memoryRequirements = VkMemoryRequirements.malloc(stack);
            vkGetBufferMemoryRequirements(device, tempBuffer, memoryRequirements);
            vkDestroyBuffer(device, tempBuffer, null);

            VkMemoryAllocateInfo allocateInfo = VkMemoryAllocateInfo.calloc(stack);
            allocateInfo.sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
            allocateInfo.allocationSize(size);
            allocateInfo.memoryTypeIndex(findMemoryType(physicalDevice, memoryRequirements.memoryTypeBits(), properties));

            if (vkAllocateMemory(device, allocateInfo, null, pBufferMemory) != VK_SUCCESS) {
                throw new RuntimeException("Failed to allocate buffer memory!");
            }

            memoryHandle = pBufferMemory.get(0);
        }
    }

    public Buffer createBuffer(long size, int usage) {
        try (MemoryStack stack = stackPush()) {
            LongBuffer pBuffer = stack.mallocLong(1);

            VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack);
            bufferInfo.sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
            bufferInfo.size(size);
            bufferInfo.usage(usage);
            bufferInfo.sharingMode(VK_SHARING_MODE_EXCLUSIVE);

            if (vkCreateBuffer(device, bufferInfo, null, pBuffer) != VK_SUCCESS) {
                throw new RuntimeException("Failed to create buffer!");
            }

            long bufferHandle = pBuffer.get(0);

            vkBindBufferMemory(device, bufferHandle, memoryHandle, totalOffset);
            buffers.put(bufferHandle, new Buffer(bufferHandle, totalOffset, size));
            long sizeInChunks = AlignmentUtils.align(size, OFFSET_ALIGNMENT);
            totalOffset += sizeInChunks;

            return buffers.get(bufferHandle);
        }
    }

    public void mapBufferMemory(Buffer buffer, PointerBuffer pData) {
        vkMapMemory(device, memoryHandle, buffer.memOffset, buffer.size, 0, pData);
    }

    public void unmapMemory() {
        vkUnmapMemory(device, memoryHandle);
    }

    @Override
    public void free() {
        buffers.forEach((handle, buffer) -> vkDestroyBuffer(device, handle, null));
        vkFreeMemory(device, memoryHandle, null);
    }

    private int findMemoryType(VkPhysicalDevice physicalDevice, int typeFilter, int properties) {
        try (VkPhysicalDeviceMemoryProperties memoryProperties = VkPhysicalDeviceMemoryProperties.malloc()) {
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, memoryProperties);

            for (int i = 0; i < memoryProperties.memoryTypeCount(); i++) {
                if ((typeFilter & (1 << i)) != 0 && (memoryProperties.memoryTypes(i).propertyFlags() & properties) == properties) {
                    return i;
                }
            }

            throw new RuntimeException("Failed to find suitable memory type!");
        }
    }
}
