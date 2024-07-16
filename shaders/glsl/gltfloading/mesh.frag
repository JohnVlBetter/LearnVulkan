#version 450

layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec4 color = texture(samplerColorMap, inUV) * vec4(inColor, 1.0);

	vec3 N = normalize(inNormal);
	vec3 L = normalize(inLightVec);
	float ndl = max(dot(N, L),0.0);
	vec3 diffuse = inColor;
	if(ndl < 0.2){
		diffuse *= vec3(0.05,0.0,0.3);
	}
	else if(ndl < 0.5){
		diffuse *= vec3(0.0,0.25,0.8);
	}else{
		diffuse *= vec3(0,0.85,0.85);
	}
	outFragColor = vec4(diffuse * color.rgb, 1.0);		
}