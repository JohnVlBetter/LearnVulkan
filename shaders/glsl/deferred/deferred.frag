#version 450

layout (binding = 1) uniform sampler2D samplerposition;
layout (binding = 2) uniform sampler2D samplerNormal;
layout (binding = 3) uniform sampler2D samplerAlbedo;
layout (binding = 4) uniform sampler2D samplerGBuffer0;
layout (binding = 5) uniform samplerCube samplerIrradiance;
layout (binding = 6) uniform sampler2D samplerBRDFLUT;
layout (binding = 7) uniform samplerCube prefilteredMap;
layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragcolor;

struct Light {
	vec4 position;
	vec3 color;
	float radius;
};

layout (binding = 8) uniform UBO 
{
	Light lights[6];
	vec4 viewPos;
	int displayDebugTarget;
} ubo;

#define PI 3.1415926535897932384626433832795
#define ALBEDO pow(texture(samplerAlbedo, inUV).rgb, vec3(2.2))

// From http://filmicgames.com/archives/75
vec3 Uncharted2Tonemap(vec3 x)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom); 
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
vec3 F_SchlickR(float cosTheta, vec3 F0, float roughness)
{
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 prefilteredReflection(vec3 R, float roughness)
{
	const float MAX_REFLECTION_LOD = 9.0; // todo: param/const
	float lod = roughness * MAX_REFLECTION_LOD;
	float lodf = floor(lod);
	float lodc = ceil(lod);
	vec3 a = textureLod(prefilteredMap, R, lodf).rgb;
	vec3 b = textureLod(prefilteredMap, R, lodc).rgb;
	return mix(a, b, lod - lodf);
}

vec3 specularContribution(vec3 L, vec3 lightColor, vec3 V, vec3 N, vec3 F0, float metallic, float roughness)
{
	// Precalculate vectors and dot products	
	vec3 H = normalize (V + L);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);

	vec3 color = vec3(0.0);

	if (dotNL > 0.0) {
		// D = Normal distribution (Distribution of the microfacets)
		float D = D_GGX(dotNH, roughness); 
		// G = Geometric shadowing term (Microfacets shadowing)
		float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
		// F = Fresnel factor (Reflectance depending on angle of incidence)
		vec3 F = F_Schlick(dotNV, F0);		
		vec3 spec = D * F * G / (4.0 * dotNL * dotNV + 0.001);		
		vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);			
		color += (kD * ALBEDO / PI + spec) * dotNL * lightColor;
	}

	return color;
}

void main() 
{
	// Get G-Buffer values
	vec3 fragPos = texture(samplerposition, inUV).rgb;
	vec4 gbuffer0 = texture(samplerGBuffer0, inUV);
	vec3 N = texture(samplerNormal, inUV).rgb;
	
	if(gbuffer0.a < 0.5)
	{
		//Skybox
		vec4 color = vec4(fragPos, 1.0);

		// Tone mapping
		color.rgb = Uncharted2Tonemap(color.rgb * 4.5f);
		color.rgb = color.rgb * (1.0f / Uncharted2Tonemap(vec3(11.2f)));	
		// Gamma correction
		color.rgb = pow(color.rgb, vec3(1.0f / 2.2f));

		outFragcolor = color;
	}
	else
	{
		//pbr
		vec3 V = normalize(ubo.viewPos.xyz - fragPos);
		vec3 R = reflect(-V, N); 

		float metallic = gbuffer0.g;
		float roughness = gbuffer0.b;

		vec3 F0 = vec3(0.04); 
		F0 = mix(F0, ALBEDO, metallic);

		vec3 Lo = vec3(0.0);

		#define lightCount 6

		for(int i = 0; i < lightCount; ++i)
		{
			vec3 L = ubo.lights[i].position.xyz - fragPos;
			vec3 lightColor = ubo.lights[i].color.rgb;
			Lo += specularContribution(L, lightColor, V, N, F0, metallic, roughness);
		}

		vec2 brdf = texture(samplerBRDFLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
		vec3 reflection = prefilteredReflection(R, roughness).rgb;	
		vec3 irradiance = texture(samplerIrradiance, N).rgb;

		// Diffuse based on irradiance
		vec3 diffuse = irradiance * ALBEDO;	

		vec3 F = F_SchlickR(max(dot(N, V), 0.0), F0, roughness);

		// Specular reflectance
		vec3 specular = reflection * (F * brdf.x + brdf.y);

		// Ambient part
		vec3 kD = 1.0 - F;
		kD *= 1.0 - metallic;	  
		vec3 ambient = (kD * diffuse + specular) * gbuffer0.rrr;

		vec3 color = ambient + Lo;

		// Tone mapping
		color = Uncharted2Tonemap(color * 4.5f);
		color = color * (1.0f / Uncharted2Tonemap(vec3(11.2f)));	
		// Gamma correction
		color = pow(color, vec3(1.0f / 2.2f));

		outFragcolor = vec4(color, texture(samplerAlbedo, inUV).a);
	}
	
	// Debug display
	if (ubo.displayDebugTarget > 0) {
		switch (ubo.displayDebugTarget) {
			case 1: 
				outFragcolor.rgb = fragPos;
				break;
			case 2: 
				outFragcolor.rgb = N;
				break;
			case 3: 
				outFragcolor.rgb = ALBEDO;
				break;
			case 4: 
				outFragcolor.rgb = gbuffer0.rrr;
				break;
			case 5: 
				outFragcolor.rgb = gbuffer0.ggg;
				break;
			case 6: 
				outFragcolor.rgb = gbuffer0.bbb;
				break;
		}		
		outFragcolor.a = 1.0;
	}
}