function SILW_Euler_1D_highorder
% 一维欧拉方程精度测试 (五阶WENO + SILW边界 + 五阶RK)
clc; clear;

gamma = 1.4;              
left = -pi; right = pi;   
t_end = 1.0;              
CFL = 0.6;             
alpha = 1.0;              

N_list = [20, 40, 80, 160, 320];
Ca_list = [0.0001, 0.9999];  
Cb = 0.7;                     

prev_errors = zeros(length(Ca_list), 2); 

fprintf('     N       Ca        L1 Error      L1 Order      L_inf Error   L_inf Order\n');
fprintf('------------------------------------------------------------------------------\n');

for iN = 1:length(N_list)
    N = N_list(iN);   
    for iCa = 1:length(Ca_list)
        Ca = Ca_list(iCa);
        dx = (right - left) / (N + Ca + Cb);
        x = left + Ca*dx + (0:N)' * dx;  
        
        U = exact_solution(x, 0, gamma);  
        
        t = 0;
        while t < t_end
            % 变量保护与波速计算
            rho_tmp = max(U(:,1), 1e-12); 
            u_tmp = U(:,2)./rho_tmp;
            p_tmp = max((gamma-1)*(U(:,3) - 0.5*rho_tmp.*u_tmp.^2), 1e-12);
            c_tmp = sqrt(gamma*p_tmp./rho_tmp);
            
            dt = CFL * dx / max(abs(u_tmp)+c_tmp);
            if t + dt > t_end, dt = t_end - t; end
            
            % ---------- 六级五阶 Runge-Kutta (RK5) ----------
            % k1 = f(Un)
            k1 = residual(U, t, dx, alpha, Ca, Cb, left, right, gamma);
            
            % k2 = f(Un + dt*1/4*k1)
            k2 = residual(U + dt*1/4*k1, t + 1/4*dt, dx, alpha, Ca, Cb, left, right, gamma);
            
            % k3 = f(Un + dt*(1/8*k1 + 1/8*k2))
            k3 = residual(U + dt*(1/8*k1 + 1/8*k2), t + 1/4*dt, dx, alpha, Ca, Cb, left, right, gamma);
            
            % k4 = f(Un + dt*(-1/2*k2 + k3))
            k4 = residual(U + dt*(-0.5*k2 + k3), t + 1/2*dt, dx, alpha, Ca, Cb, left, right, gamma);
            
            % k5 = f(Un + dt*(3/16*k1 + 9/16*k4))
            k5 = residual(U + dt*(3/16*k1 + 9/16*k4), t + 3/4*dt, dx, alpha, Ca, Cb, left, right, gamma);
            
            % k6 = f(Un + dt*(-3/7*k1 + 2/7*k2 + 12/7*k3 - 12/7*k4 + 8/7*k5))
            k6 = residual(U + dt*(-3/7*k1 + 2/7*k2 + 12/7*k3 - 12/7*k4 + 8/7*k5), t + dt, dx, alpha, Ca, Cb, left, right, gamma);
            
            % Un+1 = Un + dt/90 * (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)
            U = U + (dt/90) * (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6);
            
            t = t + dt;
        end
        
        U_exact = exact_solution(x, t_end, gamma);
        err = U(:,1) - U_exact(:,1);
        L1_err = sum(abs(err)) * dx;
        L_inf_err = max(abs(err));
        
        if iN == 1
            s_ord1 = '---'; s_ordInf = '---';
        else
            ord1 = log(prev_errors(iCa, 1)/L1_err) / log(N_list(iN)/N_list(iN-1));
            ordInf = log(prev_errors(iCa, 2)/L_inf_err) / log(N_list(iN)/N_list(iN-1));
            s_ord1 = sprintf('%.2f', ord1);
            s_ordInf = sprintf('%.2f', ordInf);
        end
        prev_errors(iCa, 1) = L1_err;
        prev_errors(iCa, 2) = L_inf_err;
        
        fprintf('%6d   %.4f   %12.4e   %8s   %12.4e   %8s\n', N, Ca, L1_err, s_ord1, L_inf_err, s_ordInf);
    end
    fprintf('------------------------------------------------------------------------------\n');
end
end

% ==================== 核心辅助函数 (由 XSimple 优化) ====================

function U = exact_solution(x, t, gamma)
    rho = 1 - 0.2 * sin(2*t - x);
    u = 2 * ones(length(x),1);
    p = 2 * ones(length(x),1);
    U = [rho, rho.*u, p/(gamma-1) + 0.5*rho.*u.^2];
end

function L = residual(U, t, dx, alpha, Ca, Cb, left, right, gamma)
    % 边界处理
    UL_g = left_boundary_SILW(U(1:5,:), t, dx, alpha, Ca, left, gamma);
    UR_g = right_boundary_robust(U(end-5:end,:)); 
    
    U_all = [UL_g(3:-1:1,:); U; UR_g];
    rho = max(U_all(:,1), 1e-12);
    m = U_all(:,2);
    E = U_all(:,3);
    u = m./rho;
    p = max((gamma-1)*(E - 0.5*rho.*u.^2), 1e-12);
    
    F_all = [m, m.*u + p, u.*(E + p)];
    c = sqrt(gamma*p./rho);
    a = max(abs(u)+c);
    
    N_pts = size(U,1);
    F_hat = zeros(size(U_all,1)-1, 3);
    for eq = 1:3
        f = F_all(:,eq); u_c = U_all(:,eq);
        fp = 0.5*(f + a*u_c); fm = 0.5*(f - a*u_c);
        F_hat(:,eq) = weno5_core(fp, 'right') + weno5_core(fm, 'left');
    end
    L = -(F_hat(4:N_pts+3,:) - F_hat(3:N_pts+2,:)) / dx;
end

function f_hat = weno5_core(v, dir)
    L = length(v); f_hat = zeros(L-1, 1); eps = 1e-10;
    if strcmp(dir, 'right')
        for i = 3:L-2
            v1=v(i-2); v2=v(i-1); v3=v(i); v4=v(i+1); v5=v(i+2);
            p0 = (2*v1 - 7*v2 + 11*v3)/6; p1 = (-v2 + 5*v3 + 2*v4)/6; p2 = (2*v3 + 5*v4 - v5)/6;
            b0 = 13/12*(v1-2*v2+v3)^2 + 1/4*(v1-4*v2+3*v3)^2;
            b1 = 13/12*(v2-2*v3+v4)^2 + 1/4*(v2-v4)^2;
            b2 = 13/12*(v3-2*v4+v5)^2 + 1/4*(3*v3-4*v4+v5)^2;
            a0 = 0.1/(b0+eps)^2; a1 = 0.6/(b1+eps)^2; a2 = 0.3/(b2+eps)^2;
            f_hat(i) = (a0*p0 + a1*p1 + a2*p2)/(a0+a1+a2);
        end
        f_hat(1)=v(1); f_hat(2)=v(2); f_hat(L-1)=v(L-1);
    else
        for i = 3:L-2
            v1=v(i+2); v2=v(i+1); v3=v(i); v4=v(i-1); v5=v(i-2);
            p0 = (2*v1 - 7*v2 + 11*v3)/6; p1 = (-v2 + 5*v3 + 2*v4)/6; p2 = (2*v3 + 5*v4 - v5)/6;
            b0 = 13/12*(v1-2*v2+v3)^2 + 1/4*(v1-4*v2+3*v3)^2;
            b1 = 13/12*(v2-2*v3+v4)^2 + 1/4*(v2-v4)^2;
            b2 = 13/12*(v3-2*v4+v5)^2 + 1/4*(3*v3-4*v4+v5)^2;
            a0 = 0.1/(b0+eps)^2; a1 = 0.6/(b1+eps)^2; a2 = 0.3/(b2+eps)^2;
            f_hat(i-1) = (a0*p0 + a1*p1 + a2*p2)/(a0+a1+a2);
        end
        f_hat(1)=v(1); f_hat(L-1)=v(L);
    end
end

function UL_ghost = left_boundary_SILW(U_int, t, dx, alpha, Ca, left, gamma)
    Ub = exact_solution(left, t, gamma);
    rho_t = -0.4 * cos(2*t - left); Ut = [rho_t, 2*rho_t, 2*rho_t]; 
    r = max(Ub(1), 1e-12); u = Ub(2)/r; 
    p = max((gamma-1)*(Ub(3) - 0.5*r*u^2), 1e-12); H = (Ub(3) + p)/r;
    A = [0, 1, 0; (gamma-3)/2*u^2, (3-gamma)*u, gamma-1; u*((gamma-1)/2*u^2 - H), H-(gamma-1)*u^2, gamma*u];
    Ux = (A \ (-Ut.')).'; 
    s_target = Ca + (1:3)'; 
    U_star = zeros(3,3);
    s_int = Ca + (0:4)';
    for c = 1:3
        coeffs_p = polyfit(s_int, U_int(:,c), 4);
        U_star(:,c) = polyval(coeffs_p, s_target);
    end
    M = [s_target.^2, s_target.^3, s_target.^4];
    b0 = Ub; b1 = Ux * dx;
    UL_ghost = zeros(3,3);
    for c = 1:3
        b_rem = M \ (U_star(:,c) - b0(c) - b1(c)*s_target);
        s_g = Ca - (1:3)';
        UL_ghost(:,c) = b0(c) + b1(c)*s_g + b_rem(1)*s_g.^2 + b_rem(2)*s_g.^3 + b_rem(3)*s_g.^4;
    end
end

function UR_ghost = right_boundary_robust(U_int)
    UR_ghost = zeros(3,3);
    s_local = (-5:0)'; s_extrap = (1:3)'; 
    for c = 1:3
        p_fit = polyfit(s_local, U_int(:,c), 5);
        UR_ghost(:,c) = polyval(p_fit, s_extrap);
    end
end