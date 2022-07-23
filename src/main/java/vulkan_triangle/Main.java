package vulkan_triangle;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.Pointer;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.joml.Vector2f;
import org.joml.Vector2fc;
import org.joml.Vector3f;
import org.joml.Vector3fc;

import static java.util.stream.Collectors.toSet;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.glfw.GLFWVulkan.glfwCreateWindowSurface;
import static org.lwjgl.glfw.GLFWVulkan.glfwGetRequiredInstanceExtensions;
import static org.lwjgl.system.Configuration.DEBUG;
import static org.lwjgl.system.MemoryStack.stackGet;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.vulkan.KHRSurface.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK10.*;
import static vulkan_triangle.ShaderSPIRVUtils.ShaderKind.FRAGMENT_SHADER;
import static vulkan_triangle.ShaderSPIRVUtils.ShaderKind.VERTEX_SHADER;
import static vulkan_triangle.ShaderSPIRVUtils.compileShaderFile;
import static vulkan_triangle.ShaderSPIRVUtils.SPIRV;

/* TODO:
 * Review ending notes of VK-Tutorial staging/index buffer sections, as they have good advice for generalizing buffers.
 * Make sure to take the offset of each buffer into account, currently they are hard-coded to be 0.
 */

public class Main {
    private static class HelloTriangleApp {
        private static class QueueFamilyIndices {
            private Integer graphicsFamily;
            private Integer presentFamily;

            private boolean isComplete() {
                return graphicsFamily != null && presentFamily != null;
            }

            private int[] unique() {
                return IntStream.of(graphicsFamily, presentFamily).distinct().toArray();
            }
        }

        private static class SwapChainSupportDetails {
            private VkSurfaceCapabilitiesKHR capabilities;
            private VkSurfaceFormatKHR.Buffer formats;
            private IntBuffer presentModes;
        }

        private static class Vertex {

            private static final int SIZEOF = (2 + 3) * Float.BYTES;
            private static final int POS_OFFSET = 0;
            private static final int COLOR_OFFSET = 2 * Float.BYTES;

            private Vector2fc pos;
            private Vector3fc color;

            public Vertex(Vector2fc pos, Vector3fc color) {
                this.pos = pos;
                this.color = color;
            }

            private static VkVertexInputBindingDescription.Buffer getBindingDescription() {

                VkVertexInputBindingDescription.Buffer bindingDescription =
                        VkVertexInputBindingDescription.calloc(1);

                bindingDescription.binding(0);
                bindingDescription.stride(Vertex.SIZEOF);
                bindingDescription.inputRate(VK_VERTEX_INPUT_RATE_VERTEX);

                return bindingDescription;
            }

            private static VkVertexInputAttributeDescription.Buffer getAttributeDescriptions() {

                VkVertexInputAttributeDescription.Buffer attributeDescriptions =
                        VkVertexInputAttributeDescription.calloc(2);

                // Position.
                VkVertexInputAttributeDescription posDescription = attributeDescriptions.get(0);
                posDescription.binding(0);
                posDescription.location(0);
                posDescription.format(VK_FORMAT_R32G32_SFLOAT);
                posDescription.offset(POS_OFFSET);

                // Color.
                VkVertexInputAttributeDescription colorDescription = attributeDescriptions.get(1);
                colorDescription.binding(0);
                colorDescription.location(1);
                colorDescription.format(VK_FORMAT_R32G32B32_SFLOAT);
                colorDescription.offset(COLOR_OFFSET);

                return attributeDescriptions.rewind();
            }

        }

        private static final Vertex[] VERTICES = {
            new Vertex(new Vector2f(-0.5f, -0.5f), new Vector3f(1.0f, 0.0f, 0.0f)),
            new Vertex(new Vector2f(0.5f, -0.5f), new Vector3f(0.0f, 1.0f, 0.0f)),
            new Vertex(new Vector2f(0.5f, 0.5f), new Vector3f(0.0f, 0.0f, 1.0f)),
            new Vertex(new Vector2f(-0.5f, 0.5f), new Vector3f(1.0f, 1.0f, 1.0f))
        };

        private static final short[] INDICES = {
                0, 1, 2,
                2, 3, 0
        };

        private static final int UINT32_MAX = 0xffffffff;
        private static final long UINT64_MAX = 0xffffffffffffffffL;

        private static final int WINDOW_WIDTH = 800;
        private static final int WINDOW_HEIGHT = 600;

        private static final int MAX_FRAMES_IN_FLIGHT = 2;

        private static final boolean ENABLE_VALIDATION_LAYERS = DEBUG.get(true);

        private static final Set<String> VALIDATION_LAYERS;

        static {
            if (ENABLE_VALIDATION_LAYERS) {
                VALIDATION_LAYERS = new HashSet<>();
                VALIDATION_LAYERS.add("VK_LAYER_KHRONOS_validation");
            } else {
                VALIDATION_LAYERS = null;
            }
        }

        private static final Set<String> DEVICE_EXTENSIONS = Stream.of(VK_KHR_SWAPCHAIN_EXTENSION_NAME).collect(toSet());

        private long window;
        private VkInstance instance;
        private long surface;
        private VkPhysicalDevice physicalDevice;
        private VkDevice device;

        private VkQueue graphicsQueue;
        private VkQueue presentQueue;

        private long swapChain;
        private List<Long> swapChainImages;
        private List<Long> swapChainImageViews;
        private List<Long> swapChainFrameBuffers;
        private int swapChainImageFormat;
        private VkExtent2D swapChainExtent;

        private long renderPass;
        private long pipelineLayout;
        private long graphicsPipeline;

        private long commandPool;

        private long vertexBuffer;
        private long vertexBufferMemory;
        private long indexBuffer;
        private long indexBufferMemory;

        private List<VkCommandBuffer> commandBuffers;

        private List<Frame> inFlightFrames;
        private Map<Integer, Frame> imagesInFlight;
        private int currentFrame;
        private boolean frameBufferResize = false;

        private SPIRV vertShaderSPIRV;
        private SPIRV fragShaderSPIRV;

        public void run() {
            initWindow();
            initVulkan();
            mainLoop();
            cleanup();
        }

        private void initWindow() {
            if (!glfwInit()) {
                throw new RuntimeException("Cannot initialize GLFW!");
            }

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

            window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "VulkanTriangle", NULL, NULL);
            glfwSetFramebufferSizeCallback(window, this::frameBufferResizeCallback);

            if (window == NULL) {
                throw new RuntimeException("Couldn't create a window!");
            }
        }

        private void frameBufferResizeCallback(long window, int width, int height) {
            frameBufferResize = true;
        }

        private void initVulkan() {
            compileShaders();

            createInstance();
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
            createCommandPool();
            createVertexBuffer();
            createIndexBuffer();
            createSwapChainObjects();
            createSyncObjects();
        }

        private void mainLoop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                drawFrame();
            }

            // Don't exit the main loop until all asynchronous operations are finished,
            // so that the resources they used can be safely cleaned up.
            vkDeviceWaitIdle(device);
        }

        private void cleanup() {
            cleanupSwapChain();

            vkDestroyBuffer(device, vertexBuffer, null);
            vkFreeMemory(device, vertexBufferMemory, null);
            vkDestroyBuffer(device, indexBuffer, null);
            vkFreeMemory(device, indexBufferMemory, null);

            inFlightFrames.forEach(frame -> {
                vkDestroySemaphore(device, frame.renderFinishedSemaphore(), null);
                vkDestroySemaphore(device, frame.imageAvailableSemaphore(), null);
                vkDestroyFence(device, frame.fence(), null);
            });
            imagesInFlight.clear();

            vkDestroyCommandPool(device, commandPool, null);

            vkDestroyDevice(device, null);
            vkDestroySurfaceKHR(instance, surface, null);
            vkDestroyInstance(instance, null);

            glfwDestroyWindow(window);
            glfwTerminate();
        }

        private void drawFrame() {
            try (MemoryStack stack = stackPush()) {
                Frame thisFrame = inFlightFrames.get(currentFrame);

                vkWaitForFences(device, thisFrame.pFence(), true, UINT64_MAX);

                IntBuffer pImageIndex = stack.mallocInt(1);

                int vkResult = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, thisFrame.imageAvailableSemaphore(), VK_NULL_HANDLE, pImageIndex);

                if (vkResult == VK_ERROR_OUT_OF_DATE_KHR) {
                    recreateSwapChain();
                    return;
                } else if (vkResult != VK_SUCCESS) {
                    throw new RuntimeException("Cannot get image!");
                }

                final int imageIndex = pImageIndex.get(0);

                if (imagesInFlight.containsKey(imageIndex)) {
                    vkWaitForFences(device, imagesInFlight.get(imageIndex).fence(), true, UINT64_MAX);
                }

                imagesInFlight.put(imageIndex, thisFrame);

                VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack);
                submitInfo.sType(VK_STRUCTURE_TYPE_SUBMIT_INFO);
                submitInfo.waitSemaphoreCount(1);
                submitInfo.pWaitSemaphores(thisFrame.pImageAvailableSemaphore());
                submitInfo.pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT));
                submitInfo.pSignalSemaphores(thisFrame.pRenderFinishedSemaphore());
                submitInfo.pCommandBuffers(stack.pointers(commandBuffers.get(imageIndex)));

                vkResetFences(device, thisFrame.pFence());

                if ((vkResult = vkQueueSubmit(graphicsQueue, submitInfo, thisFrame.fence())) != VK_SUCCESS) {
                    vkResetFences(device, thisFrame.pFence());
                    throw new RuntimeException("Failed to submit draw command buffer" + vkResult + "!");
                }

                VkPresentInfoKHR presentInfo = VkPresentInfoKHR.calloc(stack);
                presentInfo.sType(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR);
                presentInfo.pWaitSemaphores(thisFrame.pRenderFinishedSemaphore());
                presentInfo.swapchainCount(1);
                presentInfo.pSwapchains(stack.longs(swapChain));
                presentInfo.pImageIndices(pImageIndex);

                vkResult = vkQueuePresentKHR(presentQueue, presentInfo);

                if (vkResult == VK_ERROR_OUT_OF_DATE_KHR || vkResult == VK_SUBOPTIMAL_KHR || frameBufferResize) {
                    frameBufferResize = false;
                    recreateSwapChain();
                } else if (vkResult != VK_SUCCESS) {
                    throw new RuntimeException("Failed to present swap chain image!");
                }

                currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
            }
        }

        private void compileShaders() {
            vertShaderSPIRV = compileShaderFile("shaders/shader_base.vert", VERTEX_SHADER);
            fragShaderSPIRV = compileShaderFile("shaders/shader_base.frag", FRAGMENT_SHADER);
        }

        private void createIndexBuffer() {
            try (MemoryStack stack = stackPush()) {
                long bufferSize = Short.BYTES * INDICES.length;

                LongBuffer pBuffer = stack.mallocLong(1);
                LongBuffer pBufferMemory = stack.mallocLong(1);

                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pBuffer, pBufferMemory);

                long stagingBuffer = pBuffer.get(0);
                long stagingBufferMemory = pBufferMemory.get(0);

                PointerBuffer data = stack.mallocPointer(1);

                vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, data);
                {
                    memcpy(data.getByteBuffer(0, (int) bufferSize), INDICES);
                }
                vkUnmapMemory(device, stagingBufferMemory);

                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, pBuffer, pBufferMemory);

                indexBuffer = pBuffer.get(0);
                indexBufferMemory = pBufferMemory.get(0);

                copyBuffer(stagingBuffer, indexBuffer, bufferSize);

                vkDestroyBuffer(device, stagingBuffer, null);
                vkFreeMemory(device, stagingBufferMemory, null);
            }
        }

        private void createVertexBuffer() {
            try (MemoryStack stack = stackPush()) {
                long bufferSize = Vertex.SIZEOF * VERTICES.length;

                LongBuffer pBuffer = stack.mallocLong(1);
                LongBuffer pBufferMemory = stack.mallocLong(1);

                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pBuffer, pBufferMemory);

                long stagingBuffer = pBuffer.get(0);
                long stagingBufferMemory = pBufferMemory.get(0);

                PointerBuffer data = stack.mallocPointer(1);

                vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, data);
                {
                    memcpy(data.getByteBuffer(0, (int) bufferSize), VERTICES);
                }
                vkUnmapMemory(device, stagingBufferMemory);

                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, pBuffer, pBufferMemory);

                vertexBuffer = pBuffer.get(0);
                vertexBufferMemory = pBufferMemory.get(0);

                copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

                vkDestroyBuffer(device, stagingBuffer, null);
                vkFreeMemory(device, stagingBufferMemory, null);
            }
        }

        private void createBuffer(long size, int usage, int properties, LongBuffer pBuffer, LongBuffer pBufferMemory) {
            try (MemoryStack stack = stackPush()) {
                VkBufferCreateInfo bufferInfo = VkBufferCreateInfo.calloc(stack);
                bufferInfo.sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
                bufferInfo.size(size);
                bufferInfo.usage(usage);
                bufferInfo.sharingMode(VK_SHARING_MODE_EXCLUSIVE);

                if (vkCreateBuffer(device, bufferInfo, null, pBuffer) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create buffer!");
                }

                VkMemoryRequirements memoryRequirements = VkMemoryRequirements.malloc(stack);
                vkGetBufferMemoryRequirements(device, pBuffer.get(0), memoryRequirements);

                VkMemoryAllocateInfo allocateInfo = VkMemoryAllocateInfo.calloc(stack);
                allocateInfo.sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
                allocateInfo.allocationSize(memoryRequirements.size());
                allocateInfo.memoryTypeIndex(findMemoryType(memoryRequirements.memoryTypeBits(), properties));

                if (vkAllocateMemory(device, allocateInfo, null, pBufferMemory) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to allocate buffer memory!");
                }

                vkBindBufferMemory(device, pBuffer.get(0), pBufferMemory.get(0), 0);
            }
        }

        private void copyBuffer(long srcBuffer, long dstBuffer, long size) {
            try (MemoryStack stack = stackPush()) {
                VkCommandBufferAllocateInfo allocateInfo = VkCommandBufferAllocateInfo.calloc(stack);
                allocateInfo.sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
                allocateInfo.level(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
                allocateInfo.commandPool(commandPool);
                allocateInfo.commandBufferCount(1);

                PointerBuffer pCommandBuffer = stack.mallocPointer(1);
                vkAllocateCommandBuffers(device, allocateInfo, pCommandBuffer);
                VkCommandBuffer commandBuffer = new VkCommandBuffer(pCommandBuffer.get(0), device);

                VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack);
                beginInfo.sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
                beginInfo.flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

                vkBeginCommandBuffer(commandBuffer, beginInfo);
                {
                    VkBufferCopy.Buffer copyRegion = VkBufferCopy.calloc(1, stack);
                    copyRegion.size(size);
                    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, copyRegion);
                }
                vkEndCommandBuffer(commandBuffer);

                VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack);
                submitInfo.sType(VK_STRUCTURE_TYPE_SUBMIT_INFO);
                submitInfo.pCommandBuffers(pCommandBuffer);

                if (vkQueueSubmit(graphicsQueue, submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to submit copy command buffer!");
                }

                vkQueueWaitIdle(graphicsQueue);
                vkFreeCommandBuffers(device, commandPool, pCommandBuffer);
            }
        }

        private void memcpy(ByteBuffer buffer, Vertex[] vertices) {
            for (Vertex vertex : vertices) {
                buffer.putFloat(vertex.pos.x());
                buffer.putFloat(vertex.pos.y());

                buffer.putFloat(vertex.color.x());
                buffer.putFloat(vertex.color.y());
                buffer.putFloat(vertex.color.z());
            }
        }

        private void memcpy(ByteBuffer buffer, short[] indices) {
            for(short index : indices) {
                buffer.putShort(index);
            }

            buffer.rewind();
        }

        private int findMemoryType(int typeFilter, int properties) {
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

        private void createSyncObjects() {
            inFlightFrames = new ArrayList<>(MAX_FRAMES_IN_FLIGHT);
            imagesInFlight = new HashMap<>(swapChainImages.size());

            try (MemoryStack stack = stackPush()) {
                VkSemaphoreCreateInfo semaphoreInfo = VkSemaphoreCreateInfo.calloc(stack);
                semaphoreInfo.sType(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);

                VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack);
                fenceInfo.sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
                fenceInfo.flags(VK_FENCE_CREATE_SIGNALED_BIT);

                LongBuffer pImageAvailableSemaphore = stack.mallocLong(1);
                LongBuffer pRenderFinishedSemaphore = stack.mallocLong(1);
                LongBuffer pFence = stack.mallocLong(1);

                for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                    if (vkCreateSemaphore(device, semaphoreInfo, null, pImageAvailableSemaphore) != VK_SUCCESS ||
                            vkCreateSemaphore(device, semaphoreInfo, null, pRenderFinishedSemaphore) != VK_SUCCESS ||
                            vkCreateFence(device, fenceInfo, null, pFence) != VK_SUCCESS) {
                        throw new RuntimeException("Failed to create synchronization objects for the frame " + i + "!");
                    }

                    inFlightFrames.add(new Frame(pImageAvailableSemaphore.get(0), pRenderFinishedSemaphore.get(0), pFence.get(0)));
                }
            }
        }

        private void createCommandBuffers() {
            final int commandBuffersCount = swapChainFrameBuffers.size();

            commandBuffers = new ArrayList<>(commandBuffersCount);

            try (MemoryStack stack = stackPush()) {
                VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack);
                allocInfo.sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
                allocInfo.commandPool(commandPool);
                allocInfo.level(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
                allocInfo.commandBufferCount(commandBuffersCount);

                PointerBuffer pCommandBuffers = stack.mallocPointer(commandBuffersCount);

                if (vkAllocateCommandBuffers(device, allocInfo, pCommandBuffers) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to allocate command buffers!");
                }

                for (int i = 0; i < commandBuffersCount; i++) {
                    commandBuffers.add(new VkCommandBuffer(pCommandBuffers.get(i), device));
                }

                recordCommandBuffer(commandBuffersCount, stack);
            }
        }

        private void recordCommandBuffer(int commandBuffersCount, MemoryStack stack) {
            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack);
            beginInfo.sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);

            VkRenderPassBeginInfo renderPassInfo = VkRenderPassBeginInfo.calloc(stack);
            renderPassInfo.sType(VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO);
            renderPassInfo.renderPass(renderPass);

            VkRect2D renderArea = VkRect2D.calloc(stack);
            renderArea.offset(VkOffset2D.calloc(stack).set(0, 0));
            renderArea.extent(swapChainExtent);
            renderPassInfo.renderArea(renderArea);

            VkClearValue.Buffer clearValues = VkClearValue.calloc(1, stack);
            clearValues.color().float32(stack.floats(0, 0, 0, 1));
            renderPassInfo.pClearValues(clearValues);

            for (int i = 0; i < commandBuffersCount; i++) {
                VkCommandBuffer commandBuffer = commandBuffers.get(i);

                if (vkBeginCommandBuffer(commandBuffer, beginInfo) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to begin recording command buffer!");
                }

                renderPassInfo.framebuffer(swapChainFrameBuffers.get(i));

                vkCmdBeginRenderPass(commandBuffer, renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                {
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

                    LongBuffer vertexBuffers = stack.longs(vertexBuffer);
                    LongBuffer offsets = stack.longs(0);
                    vkCmdBindVertexBuffers(commandBuffer, 0, vertexBuffers, offsets);
                    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

                    vkCmdDrawIndexed(commandBuffer, INDICES.length, 1, 0, 0, 0);
                }
                vkCmdEndRenderPass(commandBuffer);

                if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to record command buffer!");
                }
            }
        }

        private void createCommandPool() {
            try (MemoryStack stack = stackPush()) {
                QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

                VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack);
                poolInfo.sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
                poolInfo.flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
                poolInfo.queueFamilyIndex(queueFamilyIndices.graphicsFamily);

                LongBuffer pCommandPool = stack.mallocLong(1);

                if (vkCreateCommandPool(device, poolInfo, null, pCommandPool) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create command pool!");
                }

                commandPool = pCommandPool.get(0);
            }
        }

        private void createFrameBuffers() {
            swapChainFrameBuffers = new ArrayList<>(swapChainImageViews.size());

            try (MemoryStack stack = stackPush()) {
                LongBuffer attachments = stack.mallocLong(1);
                LongBuffer pFrameBuffer = stack.mallocLong(1);

                VkFramebufferCreateInfo frameBufferInfo = VkFramebufferCreateInfo.calloc(stack);
                frameBufferInfo.sType(VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO);
                frameBufferInfo.renderPass(renderPass);
                frameBufferInfo.width(swapChainExtent.width());
                frameBufferInfo.height(swapChainExtent.height());
                frameBufferInfo.layers(1);

                for (long imageView : swapChainImageViews) {
                    attachments.put(0, imageView);
                    frameBufferInfo.pAttachments(attachments);

                    if (vkCreateFramebuffer(device, frameBufferInfo, null, pFrameBuffer) != VK_SUCCESS) {
                        throw new RuntimeException("Failed to create framebuffer!");
                    }

                    swapChainFrameBuffers.add(pFrameBuffer.get(0));
                }
            }
        }

        private void createRenderPass() {
            try (MemoryStack stack = stackPush()) {
                VkAttachmentDescription.Buffer colorAttachment = VkAttachmentDescription.calloc(1, stack);
                colorAttachment.format(swapChainImageFormat);
                colorAttachment.samples(VK_SAMPLE_COUNT_1_BIT);
                colorAttachment.loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR);
                colorAttachment.storeOp(VK_ATTACHMENT_STORE_OP_STORE);
                colorAttachment.stencilLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
                colorAttachment.stencilStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE);
                colorAttachment.initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
                colorAttachment.finalLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

                VkAttachmentReference.Buffer colorAttachmentRef = VkAttachmentReference.calloc(1, stack);
                colorAttachmentRef.attachment(0);
                colorAttachmentRef.layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

                VkSubpassDescription.Buffer subPass = VkSubpassDescription.calloc(1, stack);
                subPass.pipelineBindPoint(VK_PIPELINE_BIND_POINT_GRAPHICS);
                subPass.colorAttachmentCount(1);
                subPass.pColorAttachments(colorAttachmentRef);

                VkSubpassDependency.Buffer dependency = VkSubpassDependency.calloc(1, stack);
                dependency.srcSubpass(VK_SUBPASS_EXTERNAL);
                dependency.dstSubpass(0);
                dependency.srcStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
                dependency.srcAccessMask(0);
                dependency.dstStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
                dependency.dstAccessMask(VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

                VkRenderPassCreateInfo renderPassInfo = VkRenderPassCreateInfo.calloc(stack);
                renderPassInfo.sType(VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO);
                renderPassInfo.pAttachments(colorAttachment);
                renderPassInfo.pSubpasses(subPass);
                renderPassInfo.pDependencies(dependency);

                LongBuffer pRenderPass = stack.mallocLong(1);

                if (vkCreateRenderPass(device, renderPassInfo, null, pRenderPass) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create render pass!");
                }

                renderPass = pRenderPass.get(0);
            }
        }

        private void cleanupSwapChain() {
            swapChainFrameBuffers.forEach(frameBuffer -> vkDestroyFramebuffer(device, frameBuffer, null));

            vkFreeCommandBuffers(device, commandPool, asPointerBuffer(commandBuffers));

            vkDestroyPipeline(device, graphicsPipeline, null);
            vkDestroyPipelineLayout(device, pipelineLayout, null);
            vkDestroyRenderPass(device, renderPass, null);

            swapChainImageViews.forEach(imageView -> vkDestroyImageView(device, imageView, null));

            vkDestroySwapchainKHR(device, swapChain, null);
        }

        private void createSwapChainObjects() {
            createSwapChain();
            createImageViews();
            createRenderPass();
            createGraphicsPipeline();
            createFrameBuffers();
            createCommandBuffers();
        }

        private void recreateSwapChain() {
            try (MemoryStack stack = stackPush()) {
                IntBuffer width = stack.ints(0);
                IntBuffer height = stack.ints(0);

                // Pause while/if window is minimized. (Minimized windows have a size of 0).
                Runnable getWindowSize = () -> glfwGetFramebufferSize(window, width, height);
                for (getWindowSize.run(); width.get(0) == 0 || height.get(0) == 0; getWindowSize.run()) {
                    glfwWaitEvents();
                }
            }

            vkDeviceWaitIdle(device);
            cleanupSwapChain();
            createSwapChainObjects();
        }

        private void createSwapChain() {
            try (MemoryStack stack = stackPush()) {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, stack);
                VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
                int presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
                VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

                IntBuffer imageCount = stack.ints(swapChainSupport.capabilities.minImageCount() + 1);

                if (swapChainSupport.capabilities.maxImageCount() > 0 && imageCount.get(0) > swapChainSupport.capabilities.maxImageCount()) {
                    imageCount.put(0, swapChainSupport.capabilities.maxImageCount());
                }

                VkSwapchainCreateInfoKHR createInfo = VkSwapchainCreateInfoKHR.calloc(stack);
                createInfo.sType(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);
                createInfo.surface(surface);
                createInfo.minImageCount(imageCount.get(0));
                createInfo.imageFormat(surfaceFormat.format());
                createInfo.imageColorSpace(surfaceFormat.colorSpace());
                createInfo.imageExtent(extent);
                createInfo.imageArrayLayers(1);
                createInfo.imageUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

                QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

                if (!indices.graphicsFamily.equals(indices.presentFamily)) {
                    createInfo.imageSharingMode(VK_SHARING_MODE_CONCURRENT);
                    createInfo.pQueueFamilyIndices(stack.ints(indices.graphicsFamily, indices.presentFamily));
                } else {
                    createInfo.imageSharingMode(VK_SHARING_MODE_EXCLUSIVE);
                }

                createInfo.preTransform(swapChainSupport.capabilities.currentTransform());
                createInfo.compositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
                createInfo.presentMode(presentMode);
                createInfo.clipped(true);
                createInfo.oldSwapchain(VK_NULL_HANDLE);

                LongBuffer pSwapChain = stack.longs(VK_NULL_HANDLE);

                if (vkCreateSwapchainKHR(device, createInfo, null, pSwapChain) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create the swap chain!");
                }

                swapChain = pSwapChain.get(0);
                vkGetSwapchainImagesKHR(device, swapChain, imageCount, null);

                LongBuffer pSwapChainImages = stack.mallocLong(imageCount.get(0));
                vkGetSwapchainImagesKHR(device, swapChain, imageCount, pSwapChainImages);
                swapChainImages = new ArrayList<>(imageCount.get(0));

                for (int i = 0; i < pSwapChainImages.capacity(); i++) {
                    swapChainImages.add(pSwapChainImages.get(i));
                }

                swapChainImageFormat = surfaceFormat.format();
                swapChainExtent = VkExtent2D.create().set(extent);
            }
        }

        private void createImageViews() {
            swapChainImageViews = new ArrayList<>(swapChainImages.size());

            try (MemoryStack stack = stackPush()) {
                LongBuffer pImageView = stack.mallocLong(1);

                for (long swapChainImage : swapChainImages) {
                    VkImageViewCreateInfo createInfo = VkImageViewCreateInfo.calloc(stack);
                    createInfo.sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
                    createInfo.image(swapChainImage);
                    createInfo.viewType(VK_IMAGE_VIEW_TYPE_2D);
                    createInfo.format(swapChainImageFormat);

                    createInfo.components().r(VK_COMPONENT_SWIZZLE_IDENTITY);
                    createInfo.components().g(VK_COMPONENT_SWIZZLE_IDENTITY);
                    createInfo.components().b(VK_COMPONENT_SWIZZLE_IDENTITY);
                    createInfo.components().a(VK_COMPONENT_SWIZZLE_IDENTITY);

                    createInfo.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT);
                    createInfo.subresourceRange().baseMipLevel(0);
                    createInfo.subresourceRange().levelCount(1);
                    createInfo.subresourceRange().baseArrayLayer(0);
                    createInfo.subresourceRange().layerCount(1);

                    if (vkCreateImageView(device, createInfo, null, pImageView) != VK_SUCCESS) {
                        throw new RuntimeException("Failed to create image views!");
                    }

                    swapChainImageViews.add(pImageView.get(0));
                }
            }
        }

        private void createGraphicsPipeline() {
            try (MemoryStack stack = stackPush()) {
                long vertShaderModule = createShaderModule(vertShaderSPIRV.bytecode());
                long fragShaderModule = createShaderModule(fragShaderSPIRV.bytecode());

                ByteBuffer entryPoint = stack.UTF8("main");

                VkPipelineShaderStageCreateInfo.Buffer shaderStages = VkPipelineShaderStageCreateInfo.calloc(2, stack);

                VkPipelineShaderStageCreateInfo vertShaderStageInfo = shaderStages.get(0);
                vertShaderStageInfo.sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
                vertShaderStageInfo.stage(VK_SHADER_STAGE_VERTEX_BIT);
                vertShaderStageInfo.module(vertShaderModule);
                vertShaderStageInfo.pName(entryPoint);

                VkPipelineShaderStageCreateInfo fragShaderStageInfo = shaderStages.get(1);
                fragShaderStageInfo.sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
                fragShaderStageInfo.stage(VK_SHADER_STAGE_FRAGMENT_BIT);
                fragShaderStageInfo.module(fragShaderModule);
                fragShaderStageInfo.pName(entryPoint);

                // Vertex stage.
                VkPipelineVertexInputStateCreateInfo vertexInputInfo = VkPipelineVertexInputStateCreateInfo.calloc(stack);
                vertexInputInfo.sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
                vertexInputInfo.pVertexBindingDescriptions(Vertex.getBindingDescription());
                vertexInputInfo.pVertexAttributeDescriptions(Vertex.getAttributeDescriptions());

                // Assembly stage.
                VkPipelineInputAssemblyStateCreateInfo inputAssembly = VkPipelineInputAssemblyStateCreateInfo.calloc(stack);
                inputAssembly.sType(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO);
                inputAssembly.topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
                inputAssembly.primitiveRestartEnable(false);

                // Viewport and scissor.
                VkViewport.Buffer viewport = VkViewport.calloc(1, stack);
                viewport.x(0.0f);
                viewport.y(0.0f);
                viewport.width(swapChainExtent.width());
                viewport.height(swapChainExtent.height());
                viewport.minDepth(0.0f);
                viewport.maxDepth(1.0f);

                VkRect2D.Buffer scissor = VkRect2D.calloc(1, stack);
                scissor.offset(VkOffset2D.calloc(stack).set(0, 0));
                scissor.extent(swapChainExtent);

                VkPipelineViewportStateCreateInfo viewportState = VkPipelineViewportStateCreateInfo.calloc(stack);
                viewportState.sType(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO);
                viewportState.pViewports(viewport);
                viewportState.pScissors(scissor);

                // Rasterization.
                VkPipelineRasterizationStateCreateInfo rasterizer = VkPipelineRasterizationStateCreateInfo.calloc(stack);
                rasterizer.sType(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO);
                rasterizer.depthClampEnable(false);
                rasterizer.rasterizerDiscardEnable(false);
                rasterizer.polygonMode(VK_POLYGON_MODE_FILL);
                rasterizer.lineWidth(1.0f);
                rasterizer.cullMode(VK_CULL_MODE_BACK_BIT);
                rasterizer.frontFace(VK_FRONT_FACE_CLOCKWISE);
                rasterizer.depthBiasEnable(false);

                // Multi-sampling.
                VkPipelineMultisampleStateCreateInfo multisampling = VkPipelineMultisampleStateCreateInfo.calloc(stack);
                multisampling.sType(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO);
                multisampling.sampleShadingEnable(false);
                multisampling.rasterizationSamples(VK_SAMPLE_COUNT_1_BIT);

                // Color blending.
                VkPipelineColorBlendAttachmentState.Buffer colorBlendAttachment = VkPipelineColorBlendAttachmentState.calloc(1, stack);
                colorBlendAttachment.colorWriteMask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
                colorBlendAttachment.blendEnable(false);

                VkPipelineColorBlendStateCreateInfo colorBlending = VkPipelineColorBlendStateCreateInfo.calloc(stack);
                colorBlending.sType(VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO);
                colorBlending.logicOpEnable(false);
                colorBlending.logicOp(VK_LOGIC_OP_COPY);
                colorBlending.pAttachments(colorBlendAttachment);
                colorBlending.blendConstants(stack.floats(0, 0, 0, 0));

                // Pipeline layout creation.
                VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack);
                pipelineLayoutInfo.sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);

                LongBuffer pPipelineLayout = stack.longs(VK_NULL_HANDLE);

                if (vkCreatePipelineLayout(device, pipelineLayoutInfo, null, pPipelineLayout) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create pipeline layout!");
                }

                pipelineLayout = pPipelineLayout.get(0);

                // With all the previous info provided, actually create the graphics pipeline.
                VkGraphicsPipelineCreateInfo.Buffer pipelineInfo = VkGraphicsPipelineCreateInfo.calloc(1, stack);
                pipelineInfo.sType(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);
                pipelineInfo.pStages(shaderStages);
                pipelineInfo.pVertexInputState(vertexInputInfo);
                pipelineInfo.pInputAssemblyState(inputAssembly);
                pipelineInfo.pViewportState(viewportState);
                pipelineInfo.pRasterizationState(rasterizer);
                pipelineInfo.pMultisampleState(multisampling);
                pipelineInfo.pColorBlendState(colorBlending);
                pipelineInfo.layout(pipelineLayout);
                pipelineInfo.renderPass(renderPass);
                pipelineInfo.subpass(0);
                pipelineInfo.basePipelineHandle(VK_NULL_HANDLE);
                pipelineInfo.basePipelineHandle(-1);

                LongBuffer pGraphicsPipeline = stack.mallocLong(1);

                if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, pipelineInfo, null, pGraphicsPipeline) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create graphics pipeline!");
                }

                graphicsPipeline = pGraphicsPipeline.get(0);

                // Release resources.
                vkDestroyShaderModule(device, vertShaderModule, null);
                vkDestroyShaderModule(device, fragShaderModule, null);
            }
        }

        private long createShaderModule(ByteBuffer spirvCode) {
            try (MemoryStack stack = stackPush()) {
                VkShaderModuleCreateInfo createInfo = VkShaderModuleCreateInfo.calloc(stack);
                createInfo.sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
                createInfo.pCode(spirvCode);

                LongBuffer pShaderModule = stack.mallocLong(1);

                if (vkCreateShaderModule(device, createInfo, null, pShaderModule) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create shader module!");
                }

                return pShaderModule.get(0);
            }
        }

        private void createInstance() {
            if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport()) {
                throw new RuntimeException("Validation layers requested, but not available!");
            }

            try (MemoryStack stack = stackPush()) {
                VkApplicationInfo appInfo = VkApplicationInfo.calloc(stack);
                appInfo.sType(VK_STRUCTURE_TYPE_APPLICATION_INFO);
                appInfo.pApplicationName(stack.UTF8Safe("VulkanTriangle"));
                appInfo.applicationVersion(VK_MAKE_VERSION(1, 0, 0));
                appInfo.engineVersion(VK_MAKE_VERSION(1, 0, 0));
                appInfo.apiVersion(VK_API_VERSION_1_0);

                VkInstanceCreateInfo createInfo = VkInstanceCreateInfo.calloc(stack);
                createInfo.sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
                createInfo.pApplicationInfo(appInfo);
                createInfo.ppEnabledExtensionNames(glfwGetRequiredInstanceExtensions());
                createInfo.ppEnabledLayerNames(null);

                if (ENABLE_VALIDATION_LAYERS && VALIDATION_LAYERS != null) {
                    createInfo.ppEnabledLayerNames(asPointerBuffer(VALIDATION_LAYERS));
                }

                PointerBuffer instancePtr = stack.mallocPointer(1);

                if (vkCreateInstance(createInfo, null, instancePtr) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create Vulkan instance!");
                }

                instance = new VkInstance(instancePtr.get(0), createInfo);
            }
        }

        private void pickPhysicalDevice() {
            try (MemoryStack stack = stackPush()) {
                IntBuffer deviceCount = stack.ints(0);
                vkEnumeratePhysicalDevices(instance, deviceCount, null);

                if (deviceCount.get(0) == 0) {
                    throw new RuntimeException("Failed to find GPUs with Vulkan support!");
                }

                PointerBuffer ppPhysicalDevices = stack.mallocPointer(deviceCount.get(0));
                vkEnumeratePhysicalDevices(instance, deviceCount, ppPhysicalDevices);

                for (int i = 0; i < ppPhysicalDevices.capacity(); i++) {
                    VkPhysicalDevice device = new VkPhysicalDevice(ppPhysicalDevices.get(i), instance);

                    if (isDeviceSuitable(device)) {
                        physicalDevice = device;
                        return;
                    }
                }

                throw new RuntimeException("Failed to find a suitable GPU!");
            }
        }

        private boolean isDeviceSuitable(VkPhysicalDevice device) {
            QueueFamilyIndices indices = findQueueFamilies(device);
            boolean extensionsSupported = checkDeviceExtensionSupport(device);
            boolean swapChainAdequate;

            if (!extensionsSupported) return false;

            try (MemoryStack stack = stackPush()) {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, stack);
                swapChainAdequate = swapChainSupport.formats.hasRemaining() && swapChainSupport.presentModes.hasRemaining();
            }

            return indices.isComplete() && swapChainAdequate;
        }

        private boolean checkDeviceExtensionSupport(VkPhysicalDevice device) {
            try (MemoryStack stack = stackPush()) {
                IntBuffer extensionCount = stack.ints(0);
                vkEnumerateDeviceExtensionProperties(device, (String) null, extensionCount, null);

                VkExtensionProperties.Buffer availableExtensions = VkExtensionProperties.malloc(extensionCount.get(0), stack);
                vkEnumerateDeviceExtensionProperties(device, (String) null, extensionCount, availableExtensions);

                return availableExtensions.stream()
                        .map(VkExtensionProperties::extensionNameString)
                        .collect(toSet())
                        .containsAll(DEVICE_EXTENSIONS);
            }
        }

        private QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
            QueueFamilyIndices indices = new QueueFamilyIndices();

            try (MemoryStack stack = stackPush()) {
                IntBuffer queueFamilyCount = stack.ints(0);
                vkGetPhysicalDeviceQueueFamilyProperties(device, queueFamilyCount, null);

                VkQueueFamilyProperties.Buffer queueFamilies = VkQueueFamilyProperties.malloc(queueFamilyCount.get(0), stack);
                vkGetPhysicalDeviceQueueFamilyProperties(device, queueFamilyCount, queueFamilies);

                IntBuffer presentSupport = stack.ints(VK_FALSE);

                for (int i = 0; i < queueFamilies.capacity() && !indices.isComplete(); i++) {
                    if ((queueFamilies.get(i).queueFlags() & VK_QUEUE_GRAPHICS_BIT) != 0) indices.graphicsFamily = i;

                    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, presentSupport);

                    if (presentSupport.get(0) == VK_TRUE) indices.presentFamily = i;

                    i++;
                }

                return indices;
            }
        }

        private void createLogicalDevice() {
            try (MemoryStack stack = stackPush()) {
                QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

                int[] uniqueQueueFamilies = indices.unique();

                VkDeviceQueueCreateInfo.Buffer queueCreateInfos = VkDeviceQueueCreateInfo.calloc(uniqueQueueFamilies.length, stack);

                for (int i = 0; i < uniqueQueueFamilies.length; i++) {
                    VkDeviceQueueCreateInfo queueCreateInfo = queueCreateInfos.get(i);
                    queueCreateInfo.sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
                    queueCreateInfo.queueFamilyIndex(uniqueQueueFamilies[i]);
                    queueCreateInfo.pQueuePriorities(stack.floats(1.0f));
                }

                VkPhysicalDeviceFeatures deviceFeatures = VkPhysicalDeviceFeatures.calloc(stack);

                VkDeviceCreateInfo createInfo = VkDeviceCreateInfo.calloc(stack);
                createInfo.sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
                createInfo.pQueueCreateInfos(queueCreateInfos);
                createInfo.pEnabledFeatures(deviceFeatures);
                createInfo.ppEnabledExtensionNames(asPointerBuffer(DEVICE_EXTENSIONS));

                // For backwards compatibility with older Vulkan implementations, shouldn't be necessary anymore.
                if (ENABLE_VALIDATION_LAYERS && VALIDATION_LAYERS != null)
                    createInfo.ppEnabledLayerNames(asPointerBuffer(VALIDATION_LAYERS));

                PointerBuffer pDevice = stack.pointers(VK_NULL_HANDLE);

                if (vkCreateDevice(physicalDevice, createInfo, null, pDevice) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create logical device!");
                }

                device = new VkDevice(pDevice.get(0), physicalDevice, createInfo);

                PointerBuffer pQueue = stack.pointers(VK_NULL_HANDLE);
                vkGetDeviceQueue(device, indices.graphicsFamily, 0, pQueue);
                graphicsQueue = new VkQueue(pQueue.get(0), device);
                vkGetDeviceQueue(device, indices.presentFamily, 0, pQueue);
                presentQueue = new VkQueue(pQueue.get(0), device);
            }
        }

        private PointerBuffer asPointerBuffer(Collection<String> collection) {

            MemoryStack stack = stackGet();

            assert VALIDATION_LAYERS != null;
            PointerBuffer buffer = stack.mallocPointer(collection.size());

            collection.stream()
                    .map(stack::UTF8)
                    .forEach(buffer::put);

            return buffer.rewind();
        }

        private PointerBuffer asPointerBuffer(List<? extends Pointer> list) {
            MemoryStack stack = stackGet();

            PointerBuffer buffer = stack.mallocPointer(list.size());
            list.forEach(buffer::put);
            return buffer.rewind();
        }

        private boolean checkValidationLayerSupport() {
            try (MemoryStack stack = stackPush()) {
                IntBuffer layerCount = stack.ints(0);
                vkEnumerateInstanceLayerProperties(layerCount, null);
                VkLayerProperties.Buffer availableLayers = VkLayerProperties.malloc(layerCount.get(0), stack);
                vkEnumerateInstanceLayerProperties(layerCount, availableLayers);

                Set<String> availableLayerNames = availableLayers.stream()
                        .map(VkLayerProperties::layerNameString).collect(toSet());

                assert VALIDATION_LAYERS != null;
                return availableLayerNames.containsAll(VALIDATION_LAYERS);
            }
        }

        private void createSurface() {
            try (MemoryStack stack = stackPush()) {
                LongBuffer pSurface = stack.longs(VK_NULL_HANDLE);

                if (glfwCreateWindowSurface(instance, window, null, pSurface) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create window surface!");
                }

                surface = pSurface.get(0);
            }
        }

        private SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, MemoryStack stack) {
            SwapChainSupportDetails details = new SwapChainSupportDetails();

            details.capabilities = VkSurfaceCapabilitiesKHR.malloc(stack);
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, details.capabilities);

            IntBuffer count = stack.ints(0);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, count, null);

            if (count.get(0) != 0) {
                details.formats = VkSurfaceFormatKHR.malloc(count.get(0), stack);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, count, details.formats);
            }

            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, count, null);

            if (count.get(0) != 0) {
                details.presentModes = stack.mallocInt(count.get(0));
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, count, details.presentModes);
            }

            return details;
        }

        private VkSurfaceFormatKHR chooseSwapSurfaceFormat(VkSurfaceFormatKHR.Buffer availableFormats) {
            return availableFormats.stream()
                    .filter(availableFormat -> availableFormat.format() == VK_FORMAT_B8G8R8_UNORM)
                    .filter(availableFormat -> availableFormat.colorSpace() == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                    .findAny()
                    .orElse(availableFormats.get(0));
        }

        private int chooseSwapPresentMode(IntBuffer availablePresentModes) {
            for (int i = 0; i < availablePresentModes.capacity(); i++) {
                if (availablePresentModes.get(i) == VK_PRESENT_MODE_MAILBOX_KHR) {
                    return availablePresentModes.get(i);
                }
            }

            return VK_PRESENT_MODE_FIFO_KHR;
        }

        private VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR capabilities) {
            if (capabilities.currentExtent().width() != UINT32_MAX) {
                return capabilities.currentExtent();
            }

            try (VkExtent2D actualExtent = VkExtent2D.malloc().set(WINDOW_WIDTH, WINDOW_HEIGHT)) {
                VkExtent2D minExtent = capabilities.minImageExtent();
                VkExtent2D maxExtent = capabilities.maxImageExtent();

                actualExtent.width(clamp(minExtent.width(), maxExtent.width(), actualExtent.width()));
                actualExtent.height(clamp(minExtent.height(), maxExtent.height(), actualExtent.height()));

                return actualExtent;
            }
        }

        private int clamp(int min, int max, int value) {
            return Math.max(min, Math.min(max, value));
        }
    }

    public static void main(String[] args) {
        HelloTriangleApp app = new HelloTriangleApp();
        app.run();
    }
}
