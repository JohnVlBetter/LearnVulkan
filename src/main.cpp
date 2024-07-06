#include "vulkanexamplebase.h"

VulkanExampleBase *vulkanExample;
int main()
{
    vulkanExample = new VulkanExampleBase();
    vulkanExample->initWindow();
    vulkanExample->initVulkan();
    vulkanExample->renderLoop();
    delete (vulkanExample);
    return 0;
}