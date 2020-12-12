#import "ShaderLib/GLSLCompat.glsllib"
#import "ShaderLib/PBRCompat.glsllib"
#import "ShaderLib/ParallaxCompat.glsllib"
#import "ShaderLib/LightingCompat.glsllib"

varying lvec2 texCoord;
#ifdef SEPARATE_TEXCOORD
  varying lvec2 texCoord2;
#endif

varying lvec4 Color;

uniform lvec4 g_LightData[NB_LIGHTS];
uniform lvec3 g_CameraPosition;
uniform lvec4 g_AmbientLightColor;

uniform lfloat m_Roughness;
uniform lfloat m_Metallic;

varying lvec3 wPosition;


#if NB_PROBES >= 1
  uniform samplerCube g_PrefEnvMap;
  uniform lvec3 g_ShCoeffs[9];
  uniform mat4 g_LightProbeData;
#endif
#if NB_PROBES >= 2
  uniform samplerCube g_PrefEnvMap2;
  uniform lvec3 g_ShCoeffs2[9];
  uniform mat4 g_LightProbeData2;
#endif
#if NB_PROBES == 3
  uniform samplerCube g_PrefEnvMap3;
  uniform lvec3 g_ShCoeffs3[9];
  uniform mat4 g_LightProbeData3;
#endif

#ifdef BASECOLORMAP
  uniform sampler2D m_BaseColorMap;
#endif

#ifdef USE_PACKED_MR
  uniform sampler2D m_MetallicRoughnessMap;
#else
    #ifdef METALLICMAP
      uniform sampler2D m_MetallicMap;
    #endif
    #ifdef ROUGHNESSMAP
      uniform sampler2D m_RoughnessMap;
    #endif
#endif

#ifdef EMISSIVE
  uniform lvec4 m_Emissive;
#endif
#ifdef EMISSIVEMAP
  uniform sampler2D m_EmissiveMap;
#endif
#if defined(EMISSIVE) || defined(EMISSIVEMAP)
    uniform lfloat m_EmissivePower;
    uniform lfloat m_EmissiveIntensity;
#endif

#ifdef SPECGLOSSPIPELINE

  uniform lvec4 m_Specular;
  uniform lfloat m_Glossiness;
  #ifdef USE_PACKED_SG
    uniform sampler2D m_SpecularGlossinessMap;
  #else
    uniform sampler2D m_SpecularMap;
    uniform sampler2D m_GlossinessMap;
  #endif
#endif

#ifdef PARALLAXMAP
  uniform sampler2D m_ParallaxMap;
#endif
#if (defined(PARALLAXMAP) || (defined(NORMALMAP_PARALLAX) && defined(NORMALMAP)))
    uniform lfloat m_ParallaxHeight;
#endif

#ifdef LIGHTMAP
  uniform sampler2D m_LightMap;
#endif

#if defined(NORMALMAP) || defined(PARALLAXMAP)
  uniform sampler2D m_NormalMap;
  varying lvec4 wTangent;
#endif
varying lvec3 wNormal;

#ifdef DISCARD_ALPHA
  uniform lfloat m_AlphaDiscardThreshold;
#endif
//------------------------------------------------------Moblie PBR--------------------------------------------------------
// Unity3D optimizing LH
// http://filmicworlds.com/blog/optimizing-ggx-shaders-with-dotlh/
lvec2 LightingFuncGGX_FV(lfloat dotLH, lfloat roughness)
{
    lfloat alpha = roughness*roughness;

    // F
    lfloat F_a, F_b;
    lfloat dotLH5 = pow(1.0f-dotLH,5.0f);
    F_a = 1.0f;
    F_b = dotLH5;

    // V(G)
    lfloat vis;
    lfloat k = alpha * 0.5f;
    lfloat k2 = k*k;
    lfloat invK2 = 1.0f-k2;
    vis = 1.0f / (dotLH*dotLH*invK2 + k2);

    return lvec2(F_a*vis,F_b*vis);
}
#define invPI       0.3183098861837697f
#define invTWO_PI   0.15915494309f
#define saturate(x) clamp(x, 0.0f, 1.0f)
#define specularAttn 0.25f


/*
 * Standard Classic BRDF
 */
void Classic_PBR_ComputeDirectLight(lvec3 normal, lvec3 lightDir, lvec3 viewDir,
                            lvec3 lightColor, lfloat fZero, lfloat roughness, lfloat ndotv,
                            out lvec3 outDiffuse, out lvec3 outSpecular){
    // Compute halfway vector.
    lvec3 halfVec = normalize(lightDir + viewDir);

    // Compute ndotl, ndoth,  vdoth terms which are needed later.
    lfloat ndotl = max( dot(normal,   lightDir), 0.0f);
    lfloat ndoth = max( dot(normal,   halfVec),  0.0f);
    lfloat hdotv = max( dot(viewDir,  halfVec),  0.0f);


    // Compute diffuse using energy-conserving Lambert.
    // Alternatively, use Oren-Nayar for really rough
    // materials or if you have lots of processing power ...
    outDiffuse = lvec3(ndotl) * lightColor;

    //cook-torrence, microfacet BRDF : http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

    lfloat alpha = roughness * roughness;

    //D, GGX normaal Distribution function
    lfloat alpha2 = alpha * alpha;
    lfloat sum  = ((ndoth * ndoth) * (alpha2 - 1.0f) + 1.0f);
    lfloat denom = PI * sum * sum;
    lfloat D = alpha2 / denom;

/*
    // F,G
    lfloat ldoth = max( dot(lightDir, halfVec),  0.0f);
    lvec2 FV_helper = LightingFuncGGX_FV(ldoth,roughness);
    lfloat FV = fZero*FV_helper.x + (1.0f-fZero)*FV_helper.y;
    lfloat specular = ldoth * D * FV;
*/

    // Compute Fresnel function via Schlick's approximation.
    lfloat fresnel = fZero + ( 1.0f - fZero ) * pow( 2.0f, (-5.55473f * hdotv - 6.98316f) * hdotv);


    //G Shchlick GGX Gometry shadowing term,  k = alpha/2
    lfloat k = alpha * 0.5f;

    //classic Schlick ggx
    lfloat G_V = ndotv / (ndotv * (1.0f - k) + k);
    lfloat G_L = ndotl / (ndotl * (1.0f - k) + k);
    lfloat G = ( G_V * G_L );

    lfloat specular =(D* fresnel * G) /(4.0f * ndotv);





 /*
    // UE4 way to optimise shlick GGX Gometry shadowing term
    //http://graphicrants.blogspot.co.uk/2013/08/specular-brdf-reference.html
    lfloat G_V = ndotv + sqrt( (ndotv - ndotv * k) * ndotv + k );
    lfloat G_L = ndotl + sqrt( (ndotl - ndotl * k) * ndotl + k );
    // the max here is to avoid division by 0 that may cause some small glitches.
    lfloat G = 1.0f/max( G_V * G_L ,0.01f);

    lfloat specular = D * fresnel * G * ndotl;
    */


    outSpecular = lvec3(specular) * lightColor;
}
lvec3 fresnelSchlickRoughness(lfloat cosTheta, lvec3 F0, lfloat roughness)
{
    return F0 + (max(vec3(1.0f - roughness), F0) - F0) * pow(1.0f - cosTheta, 5.0f);
}

lvec2 getuv(lvec3 p){
    lfloat theta = acos(p.y);
    lfloat phi = atan(p.z, p.x);
    if (phi < 0.0f) {
        phi += 2.0f * PI;
    }
    lvec2 s;
    s.x = phi * invTWO_PI;
    s.y = theta * invPI;
    return s;
}

// unreal4 mobile
// direct diffuseBRDF( use lamberPi )
lfloat LamberPi( lfloat ndotl ){
    // ndot1 / PI
    return ndotl * invPI;
}
// direct specularBRDF
// D,NormalizedBlinnPhong(NDF)
// The Blinn-Phong Normalization Zoo
// http://www.thetenthplanet.de/archives/255
lfloat D_NormalizedBlinnPhong( lfloat ndoth, lfloat specularRoughness ){
    // D = ( n + 1 ) / 2Π * ( N . H )^n;
    // n is the specular exponent.
    lfloat distribution = pow( ndoth, 1.0f - specularRoughness );
    // distribution *= ( 1 + specularGloss ) / 2PI;specularGloss = 1.0f - specularRoughness
    distribution *= (2.0f - specularRoughness) * invTWO_PI;
    return distribution;
}
// D,Approx
// https://www.unrealengine.com/zh-CN/blog/physically-based-shading-on-mobile
lfloat D_Approx( lfloat roughness, lfloat rdotl ){
    lfloat a = roughness * roughness;
    lfloat a2 = a * a;
    lfloat rcp_a2 = 1.0f / a2;
    lfloat c = 0.72134752f * rcp_a2 + 0.39674113f;
    return rcp_a2 * exp2( c * rdotl - c );
}
// F,V(F,G)
// The Blinn-Phong Normalization Zoo
// http://www.thetenthplanet.de/archives/255
lfloat FV_NormalizedBlinnPhong( lfloat ldoth ){
    // pow( ldoth, -3.0f ) / 4.0f
    return pow( ldoth, -3.0f ) / 4.0f;
}
// direct Specular BRDF
// The Blinn-Phong Normalization Zoo
// http://www.thetenthplanet.de/archives/255
lvec3 NormalizedBlinnPhongBRDF( lfloat ndoth, lfloat ldoth, lfloat specularRoughness, lvec3 specularColor ){
    //lfloat D = D_NormalizedBlinnPhong( ndoth, specularRoughness );
    lfloat D = D_Approx( specularRoughness, ndoth );
    lfloat FV = FV_NormalizedBlinnPhong( ldoth );
    return D * FV * specularColor;
}
// indirectSpecular BRDF
// https://www.unrealengine.com/zh-CN/blog/physically-based-shading-on-mobile
lvec3 IndirectMobileEnvBRDFApprox(lvec3 specularColor, lfloat ndotv, lfloat roughness){
    const lvec4 c0 = lvec4( -1.0f, -0.0275f, -0.572f, 0.022f );
    const lvec4 c1 = lvec4( 1.0f, 0.0425f, 1.04f, -0.04f );
    lvec4 r = roughness * c0 + c1;
    lfloat a004 = min( r.x * r.x, exp2( -9.28f * ndotv ) ) * r.x + r.y;
    lvec2 AB = lvec2( -1.04f, 1.04f ) * a004 + r.zw;
    lvec3 F_L = specularColor * AB.x + AB.y;
    return F_L;
}
// use Approx Specular BRDF
void Mobile_PBR_ComputeDirectLightOp1(lvec3 normal, lvec3 lightDir, lvec3 viewDir,
                           lvec3 lightColor, lfloat roughness,
                           out lvec3 outDiffuse, out lvec3 outSpecular){
    lvec3 halfVec = normalize(lightDir + viewDir);
    lfloat ndotl = max( dot(normal,   lightDir), 0.0f);
    lfloat ndoth = max( dot(normal,   halfVec),  0.0f);
    // diffuse Lambert BRDF
    outDiffuse = lvec3(ndotl) * lightColor;

    //D, GGX normaal Distribution function(NDF)
    lfloat alpha = roughness * roughness;
    lfloat alpha2 = alpha * alpha;
    lfloat sum  = ((ndoth * ndoth) * (alpha2 - 1.0f) + 1.0f);
    lfloat denom = PI * sum * sum;
    lfloat D = alpha2 / denom;

    /*
    //D,BlinnPhong
    lfloat D = pow( ndoth, 1.0f - roughness );
    // distribution *= ( 1 + specularGloss ) / 2PI;specularGloss = 1.0f - specularRoughness
    D *= (2.0f - roughness) * invTWO_PI;
    */

    /*
    //D, GGX/Trowbridge-Reitz
    lfloat alpha = roughness * roughness;
    lfloat n = ndoth * alpha;
    lfloat p = alpha / (ndoth * ndoth + n * n);
    lfloat D = p * p;
    */
    // specular BRDF(f(x) = (roughness * 0.25f + 0.25f) * D
    lfloat specular = (roughness * specularAttn + specularAttn) * D;
    outSpecular = lvec3(specular) * lightColor;
}
// use NormalizedBlinnPhongBRDF
void Mobile_PBR_ComputeDirectLightOp2(lvec3 normal, lvec3 lightDir, lvec3 viewDir,
                         lvec3 specularColor, lfloat fZero, lfloat roughness, lfloat metallic, lfloat ndotv,
                         out lvec3 outDiffuse, out lvec3 outSpecular){
    // Compute halfway vector.
    lvec3 halfVec = normalize(lightDir + viewDir);

    // Compute ndotl, ndoth,  vdoth terms which are needed later.
    lfloat ndotl = max( dot(normal,   lightDir), 0.0f);
    lfloat ndoth = max( dot(normal,   halfVec),  0.0f);
    lfloat hdotv = max( dot(viewDir,  halfVec),  0.0f);
    lfloat ldoth = max( dot(lightDir,  halfVec),  0.0f);

    lfloat fresnel = fZero + ( 1.0f - fZero ) * pow( 2.0f, (-5.55473f * hdotv - 6.98316f) * hdotv);
    lfloat kd = (1.0f - fresnel ) * ( 1.0f - metallic );
    outDiffuse = lvec3( LamberPi( ndotl ) * kd );
    lvec3 specularBRDF = NormalizedBlinnPhongBRDF( ndoth, ldoth, roughness, specularColor);
    outSpecular = specularBRDF;
}
//------------------------------------------------------Moblie PBR--------------------------------------------------------

void main(){
    lvec2 newTexCoord;
    lvec3 viewDir = normalize(g_CameraPosition - wPosition);

    lvec3 norm = normalize(wNormal);
    #if defined(NORMALMAP) || defined(PARALLAXMAP)
        lvec3 tan = normalize(wTangent.xyz);
        mat3 tbnMat = mat3(tan, wTangent.w * cross( (norm), (tan)), norm);
    #endif

    #if (defined(PARALLAXMAP) || (defined(NORMALMAP_PARALLAX) && defined(NORMALMAP)))
       lvec3 vViewDir =  viewDir * tbnMat;
       #ifdef STEEP_PARALLAX
           #ifdef NORMALMAP_PARALLAX
               //parallax map is stored in the alpha channel of the normal map
               newTexCoord = steepParallaxOffset(m_NormalMap, vViewDir, texCoord, m_ParallaxHeight);
           #else
               //parallax map is a texture
               newTexCoord = steepParallaxOffset(m_ParallaxMap, vViewDir, texCoord, m_ParallaxHeight);
           #endif
       #else
           #ifdef NORMALMAP_PARALLAX
               //parallax map is stored in the alpha channel of the normal map
               newTexCoord = classicParallaxOffset(m_NormalMap, vViewDir, texCoord, m_ParallaxHeight);
           #else
               //parallax map is a texture
               newTexCoord = classicParallaxOffset(m_ParallaxMap, vViewDir, texCoord, m_ParallaxHeight);
           #endif
       #endif
    #else
       newTexCoord = texCoord;
    #endif

    #ifdef BASECOLORMAP
        lvec4 albedo = texture2D(m_BaseColorMap, newTexCoord) * Color;
    #else
        lvec4 albedo = Color;
    #endif

    #ifdef USE_PACKED_MR
        lvec2 rm = texture2D(m_MetallicRoughnessMap, newTexCoord).gb;
        lfloat Roughness = rm.x * max(m_Roughness, 1e-4);
        lfloat Metallic = rm.y * max(m_Metallic, 0.0);
    #else
        #ifdef ROUGHNESSMAP
            lfloat Roughness = texture2D(m_RoughnessMap, newTexCoord).r * max(m_Roughness, 1e-4);
        #else
            lfloat Roughness =  max(m_Roughness, 1e-4);
        #endif
        #ifdef METALLICMAP
            lfloat Metallic = texture2D(m_MetallicMap, newTexCoord).r * max(m_Metallic, 0.0);
        #else
            lfloat Metallic =  max(m_Metallic, 0.0);
        #endif
    #endif

    lfloat alpha = albedo.a;

    #ifdef DISCARD_ALPHA
        if(alpha < m_AlphaDiscardThreshold){
            discard;
        }
    #endif

    // ***********************
    // Read from textures
    // ***********************
    #if defined(NORMALMAP)
      lvec4 normalHeight = texture2D(m_NormalMap, newTexCoord);
      //Note the -2.0 and -1.0. We invert the green channel of the normal map,
      //as it's complient with normal maps generated with blender.
      //see http://hub.jmonkeyengine.org/forum/topic/parallax-mapping-fundamental-bug/#post-256898
      //for more explanation.
      lvec3 normal = normalize((normalHeight.xyz * lvec3(2.0f, NORMAL_TYPE * 2.0f, 2.0f) - lvec3(1.0f, NORMAL_TYPE * 1.0f, 1.0f)));
      normal = normalize(tbnMat * normal);
      //normal = normalize(normal * inverse(tbnMat));
    #else
      lvec3 normal = norm;
    #endif

    #ifdef SPECGLOSSPIPELINE

        #ifdef USE_PACKED_SG
            lvec4 specularColor = texture2D(m_SpecularGlossinessMap, newTexCoord);
            lfloat glossiness = specularColor.a * m_Glossiness;
            specularColor *= m_Specular;
        #else
            #ifdef SPECULARMAP
                lvec4 specularColor = texture2D(m_SpecularMap, newTexCoord);
            #else
                lvec4 specularColor = lvec4(1.0);
            #endif
            #ifdef GLOSSINESSMAP
                lfloat glossiness = texture2D(m_GlossinessMap, newTexCoord).r * m_Glossiness;
            #else
                lfloat glossiness = m_Glossiness;
            #endif
            specularColor *= m_Specular;
        #endif
        lvec4 diffuseColor = albedo;// * (1.0 - max(max(specularColor.r, specularColor.g), specularColor.b));
        Roughness = 1.0 - glossiness;
        lvec3 fZero = specularColor.xyz;
    #else
        lfloat specular = 0.5;
        lfloat nonMetalSpec = 0.08 * specular;
        lvec4 specularColor = (nonMetalSpec - nonMetalSpec * Metallic) + albedo * Metallic;
        lvec4 diffuseColor = albedo - albedo * Metallic;
        lvec3 fZero = lvec3(specular);
    #endif

    gl_FragColor.rgb = lvec3(0.0);
    lvec3 ao = lvec3(1.0);

    #ifdef LIGHTMAP
       lvec3 lightMapColor;
       #ifdef SEPARATE_TEXCOORD
          lightMapColor = texture2D(m_LightMap, texCoord2).rgb;
       #else
          lightMapColor = texture2D(m_LightMap, texCoord).rgb;
       #endif
       #ifdef AO_MAP
         lightMapColor.gb = lightMapColor.rr;
         ao = lightMapColor;
       #else
         gl_FragColor.rgb += diffuseColor.rgb * lightMapColor;
       #endif
       specularColor.rgb *= lightMapColor;
    #endif


    lfloat ndotv = max( dot( normal, viewDir ),0.0);
    for( int i = 0;i < NB_LIGHTS; i+=3){
        lvec4 lightColor = g_LightData[i];
        lvec4 lightData1 = g_LightData[i+1];
        lvec4 lightDir;
        lvec3 lightVec;
        lightComputeDir(wPosition, lightColor.w, lightData1, lightDir, lightVec);

        lfloat fallOff = 1.0;
        #if __VERSION__ >= 110
            // allow use of control flow
        if(lightColor.w > 1.0){
        #endif
            fallOff =  computeSpotFalloff(g_LightData[i+2], lightVec);
        #if __VERSION__ >= 110
        }
        #endif
        //point light attenuation
        fallOff *= lightDir.w;

        lightDir.xyz = normalize(lightDir.xyz);
        lvec3 directDiffuse;
        lvec3 directSpecular;

        #if ( defined( PATTERN1 ) || defined( PATTERN2 ) )
            Mobile_PBR_ComputeDirectLightOp1(normal, lightDir.xyz, viewDir,
                             lightColor.rgb, Roughness,
                             directDiffuse,  directSpecular);

            lvec3 directLighting = diffuseColor.rgb *directDiffuse + directSpecular;
        #else
            #if ( defined( PATTERN3 ) || defined( PATTERN4 ) )
                Mobile_PBR_ComputeDirectLightOp2(normal, lightDir.xyz, viewDir,
                                    specularColor.rgb, fZero.x, Roughness, Metallic, ndotv,
                                    directDiffuse,  directSpecular);
                lvec3 directLighting = ( diffuseColor.rgb * directDiffuse + directSpecular ) * lightColor.rgb;
            #else
                Classic_PBR_ComputeDirectLight(normal, lightDir.xyz, viewDir,
                            lightColor.rgb, fZero.x, Roughness, ndotv,
                            directDiffuse,  directSpecular);
                lvec3 directLighting = diffuseColor.rgb *directDiffuse + directSpecular;
            #endif
        #endif

        gl_FragColor.rgb += directLighting * fallOff;
    }

    /*
    #if NB_PROBES >= 1
        lvec3 color1 = lvec3(0.0);
        lvec3 color2 = lvec3(0.0);
        lvec3 color3 = lvec3(0.0);
        lfloat weight1 = 1.0;
        lfloat weight2 = 0.0;
        lfloat weight3 = 0.0;

        lfloat ndf = renderProbe(viewDir, wPosition, normal, norm, Roughness, diffuseColor, specularColor, ndotv, ao, g_LightProbeData, g_ShCoeffs, g_PrefEnvMap, color1);
        #if NB_PROBES >= 2
            lfloat ndf2 = renderProbe(viewDir, wPosition, normal, norm, Roughness, diffuseColor, specularColor, ndotv, ao, g_LightProbeData2, g_ShCoeffs2, g_PrefEnvMap2, color2);
        #endif
        #if NB_PROBES == 3
            lfloat ndf3 = renderProbe(viewDir, wPosition, normal, norm, Roughness, diffuseColor, specularColor, ndotv, ao, g_LightProbeData3, g_ShCoeffs3, g_PrefEnvMap3, color3);
        #endif

        #if NB_PROBES >= 2
            lfloat invNdf =  max(1.0 - ndf,0.0);
            lfloat invNdf2 =  max(1.0 - ndf2,0.0);
            lfloat sumNdf = ndf + ndf2;
            lfloat sumInvNdf = invNdf + invNdf2;
            #if NB_PROBES == 3
                lfloat invNdf3 = max(1.0 - ndf3,0.0);
                sumNdf += ndf3;
                sumInvNdf += invNdf3;
                weight3 =  ((1.0 - (ndf3 / sumNdf)) / (NB_PROBES - 1)) *  (invNdf3 / sumInvNdf);
            #endif

            weight1 = ((1.0 - (ndf / sumNdf)) / (NB_PROBES - 1)) *  (invNdf / sumInvNdf);
            weight2 = ((1.0 - (ndf2 / sumNdf)) / (NB_PROBES - 1)) *  (invNdf2 / sumInvNdf);

            lfloat weightSum = weight1 + weight2 + weight3;

            weight1 /= weightSum;
            weight2 /= weightSum;
            weight3 /= weightSum;
        #endif

        #ifdef USE_AMBIENT_LIGHT
            color1.rgb *= g_AmbientLightColor.rgb;
            color2.rgb *= g_AmbientLightColor.rgb;
            color3.rgb *= g_AmbientLightColor.rgb;
        #endif
        gl_FragColor.rgb += color1 * clamp(weight1,0.0,1.0) + color2 * clamp(weight2,0.0,1.0) + color3 * clamp(weight3,0.0,1.0);

    #endif
    */

/*
    #if NB_PROBES >= 1
        lvec3 rv = reflect(-viewDir.xyz, normal.xyz);
        //prallax fix for spherical bounds from https://seblagarde.wordpress.com/2012/09/29/image-based-lighting-approaches-and-parallax-corrected-cubemap/
        // g_LightProbeData.w is 1/probe radius + nbMipMaps, g_LightProbeData.xyz is the position of the lightProbe.
        lfloat invRadius = fract( g_LightProbeData[3].w);
        lfloat nbMipMaps = g_LightProbeData[3].w - invRadius;
        rv = invRadius * (wPosition - g_LightProbeData[3].xyz) +rv;

         //horizon fade from http://marmosetco.tumblr.com/post/81245981087
        lfloat horiz = dot(rv, norm);
        lfloat horizFadePower = 1.0f - Roughness;
        horiz = clamp( 1.0f + horizFadePower * horiz, 0.0f, 1.0f );
        horiz *= horiz;

        lvec3 indirectDiffuse = lvec3(0.0f);
        lvec3 indirectSpecular = lvec3(0.0f);
        //
        indirectDiffuse = sphericalHarmonics(normal.xyz, g_ShCoeffs) * diffuseColor.rgb;
        lvec3 dominantR = getSpecularDominantDir( normal, rv.xyz, Roughness*Roughness );
        indirectSpecular = ApproximateSpecularIBLPolynomial(g_PrefEnvMap, specularColor.rgb, Roughness, ndotv, dominantR, nbMipMaps);
        indirectSpecular *= lvec3(horiz);

        lvec3 indirectLighting = (indirectDiffuse + indirectSpecular) * ao;

        gl_FragColor.rgb = gl_FragColor.rgb + indirectLighting * step( 0.0f, g_LightProbeData[3].w);
    #endif
    */
/*
    // LearnGL IBL

    #if NB_PROBES >= 1
        lvec3 F = fresnelSchlickRoughness(ndotv, fZero, Roughness);

        lvec3 kS = F;
        lvec3 kD = 1.0f - kS;
        kD *= 1.0f - Metallic;

        //lvec3 irradiance = texture2D(m_BaseColorMap, newTexCoord).rgb;
        //lvec3 indirectDiffuse      = irradiance * albedo.rgb;
        lvec3 indirectDiffuse = sphericalHarmonics(normal.xyz, g_ShCoeffs) * diffuseColor.rgb;

        // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
        lvec3 rv = reflect(-viewDir.xyz, normal.xyz);
        const lfloat MAX_REFLECTION_LOD = 4.0f;
        lvec3 prefilteredColor = textureCubeLod(g_PrefEnvMap, rv,  Roughness * MAX_REFLECTION_LOD).rgb;
        lvec2 brdf  = texture2D(m_BaseColorMap, vec2(ndotv, Roughness)).rg;
        lvec3 indirectSpecular = prefilteredColor * (F * brdf.x + brdf.y);

        lvec3 indirectLighting = (kD * indirectDiffuse) * ao;
        gl_FragColor.rgb = gl_FragColor.rgb + indirectLighting;
    #endif
    */
    /*
    #if NB_PROBES >= 1
        //lvec3 rv = normal;
        const lfloat brdf = 0.25f;
        lvec3 indirectSpecular = textureCubeLod(g_PrefEnvMap, normal, Roughness * 4.0f).rgb;
        //lvec3 indirectSpecular = lvec3(ao * 0.25f);
        lvec3 indirectLighting = (diffuseColor.rgb + indirectSpecular) * brdf * ao;
        gl_FragColor.rgb = gl_FragColor.rgb + indirectLighting * step( 0.0f, g_LightProbeData[3].w);
    #endif
    */
    // mobile IndirectLight
    /*
    lvec3 indirectDiffuse = diffuseColor.rgb;
    lvec3 indirectSpecular = specularColor.rgb;
    lvec3 envBRDFApprox = IndirectMobileEnvBRDFApprox( specularColor.rgb, ndotv, Roughness );
    lvec3 indirectLighting = indirectDiffuse * ao + indirectSpecular * envBRDFApprox;
    gl_FragColor.rgb = gl_FragColor.rgb + indirectLighting;
    */
    #if ( defined( PATTERN1 ) || defined( PATTERN3 ) )
        // ApproxEnvBRDF
        lvec3 envBRDFApprox = IndirectMobileEnvBRDFApprox( lvec3( Metallic ), ndotv, Roughness );
        lvec3 indirectLighting = diffuseColor.rgb * ao + specularColor.rgb * envBRDFApprox;
        gl_FragColor.rgb = gl_FragColor.rgb + indirectLighting;
    #else
        #if ( defined( PATTERN2 ) || defined( PATTERN4 ) )
            #if NB_PROBES >= 1
                // ue4 mobile IBL
                // specualr BRDF:
                // When there is Lightmap, use BRDF = Lightmap brightness
                // When PointLIghtIndirectCache exists, BRDF = the brightness of PointLihgtIndirect
                // When neither of the above exists, use BRDF = 1
                // Approx IBL
                // IBL = SamplerCube(Cubemap ,Roughness ) * BRDF * MaterialAO
                //lvec3 rv = normal;
                lvec3 rv = reflect(-viewDir.xyz, normal.xyz);
                //prallax fix for spherical bounds from https://seblagarde.wordpress.com/2012/09/29/image-based-lighting-approaches-and-parallax-corrected-cubemap/
                // g_LightProbeData.w is 1/probe radius + nbMipMaps, g_LightProbeData.xyz is the position of the lightProbe.
                const lfloat brdf = 1.0f;
                lvec3 indirectSpecular = textureCubeLod(g_PrefEnvMap, rv, Roughness * 4.0f).rgb;
                //lvec3 indirectSpecular = lvec3(ao * 0.25f);
                lvec3 indirectLighting = (diffuseColor.rgb + indirectSpecular * specularColor.rgb) * brdf * ao;
                gl_FragColor.rgb = gl_FragColor.rgb + indirectLighting * step( 0.0f, g_LightProbeData[3].w);
            #endif
        #else
            #if NB_PROBES >= 1
                lvec3 rv = reflect(-viewDir.xyz, normal.xyz);
                //prallax fix for spherical bounds from https://seblagarde.wordpress.com/2012/09/29/image-based-lighting-approaches-and-parallax-corrected-cubemap/
                // g_LightProbeData.w is 1/probe radius + nbMipMaps, g_LightProbeData.xyz is the position of the lightProbe.
                lfloat invRadius = fract( g_LightProbeData[3].w);
                lfloat nbMipMaps = g_LightProbeData[3].w - invRadius;
                rv = invRadius * (wPosition - g_LightProbeData[3].xyz) +rv;

                 //horizon fade from http://marmosetco.tumblr.com/post/81245981087
                lfloat horiz = dot(rv, norm);
                lfloat horizFadePower = 1.0f - Roughness;
                horiz = clamp( 1.0f + horizFadePower * horiz, 0.0f, 1.0f );
                horiz *= horiz;

                lvec3 indirectDiffuse = lvec3(0.0f);
                lvec3 indirectSpecular = lvec3(0.0f);
                //
                indirectDiffuse = sphericalHarmonics(normal.xyz, g_ShCoeffs) * diffuseColor.rgb;
                lvec3 dominantR = getSpecularDominantDir( normal, rv.xyz, Roughness*Roughness );
                indirectSpecular = ApproximateSpecularIBLPolynomial(g_PrefEnvMap, specularColor.rgb, Roughness, ndotv, dominantR, nbMipMaps);
                indirectSpecular *= lvec3(horiz);

                lvec3 indirectLighting = (indirectDiffuse + indirectSpecular) * ao;

                gl_FragColor.rgb = gl_FragColor.rgb + indirectLighting * step( 0.0f, g_LightProbeData[3].w);
            #endif
        #endif
    #endif
    

    #if defined(EMISSIVE) || defined (EMISSIVEMAP)
        #ifdef EMISSIVEMAP
            lvec4 emissive = texture2D(m_EmissiveMap, newTexCoord);
        #else
            lvec4 emissive = m_Emissive;
        #endif
        gl_FragColor += emissive * pow(emissive.a, m_EmissivePower) * m_EmissiveIntensity;
    #endif
    gl_FragColor.a = alpha;
    // HDR tonemapping
    gl_FragColor.rgb = gl_FragColor.rgb / (gl_FragColor.rgb + lvec3(1.0f));
    // gamma correct
    //gl_FragColor.rgb = pow(gl_FragColor.rgb, lvec3(1.0f/2.2f));
   
}
