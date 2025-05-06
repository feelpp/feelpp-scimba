SetFactory("OpenCASCADE");
    h=0.0125;
    dim=2;
    
        Disk(1) = {0, 0, 0, 1.0};        
        Characteristic Length{ PointsOf{ Surface{1}; } } = h;
        Physical Curve("Gamma_D") = {1};
        Physical Surface("Omega") = {1};
        