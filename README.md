# MobilePBRLighting
PBR lighting optimization for mobile
usage:
After downloading, put it into your project, select MobilePBRLighting.j3md material for the object, and then in the material parameters, there are 4 options, divided into Pattern1 (approximate mirror + approximate IndirectLight), Pattern2 (approximate mirror + approximate IBL) , Pattern3 (NormalizedBlinnPhong + approximately IndirectLight) and Pattern4 (NormalizedBlinnPhong + approximately IBL), the default value is Pattern1,
If you do not select any of them, it is the standard PBR lighting effect.
In addition, if necessary, you can replace and modify the default PBR.glsllib, and then integrate it into the default PBRLighting.j3md, or you may need to write your own code to replace PBRLighting.j3md with MobilePBRLighting.j3md for all geometric objects.
