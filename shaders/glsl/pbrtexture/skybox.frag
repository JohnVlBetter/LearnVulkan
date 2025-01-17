#version 450

layout (binding = 2) uniform samplerCube samplerEnv;

layout (location = 0) in vec3 inUVW;

layout (location = 0) out vec4 outPosition;
//layout (location = 1) out vec4 outNormal;
//layout (location = 2) out vec4 outAlbedo;
//layout (location = 3) out vec4 outGBuffer0;

layout (binding = 1) uniform UBOParams {
	vec4 lights[4];
	float exposure;
	float gamma;
} uboParams;

// From http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 Uncharted2Tonemap(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

void main() 
{
	vec3 color = texture(samplerEnv, inUVW).rgb;

	// Tone mapping
	//color = Uncharted2Tonemap(color * uboParams.exposure);
	//color = color * (1.0f / Uncharted2Tonemap(vec3(11.2f)));	
	// Gamma correction
	//color = pow(color, vec3(1.0f / uboParams.gamma));
	
	outPosition = vec4(color, 1.0);
	//outNormal = vec4(0.0, 0.0, 1.0, 1.0);
	//outAlbedo = vec4(color, 1.0);
	//outGBuffer0 = vec4(0.0, 0.0, 0.0, 0.0);
}