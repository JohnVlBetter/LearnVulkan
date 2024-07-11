#version 450

layout (binding = 1) uniform sampler2DArray samplerArray;

layout (location = 0) in vec2 inUV;
layout (location = 1) in float inIdx;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	outFragColor = texture(samplerArray, vec3(inUV, inIdx));
}