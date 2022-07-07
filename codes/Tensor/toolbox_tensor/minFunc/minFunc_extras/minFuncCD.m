function [x] = minFuncCD(obj,x,options)

% Select
CYCLIC = 0;
RANDOM = 1;
LIPSCHITZ = 2;
RANDOM_MULTI = 3;
GREEDY = 4;

% Initial Step Length
PREVIOUS = 0;
NEWTON = 1;

% Interpolation
BISECT = 0;
POLY = 1;

% Search Method
ARMIJO = 0;
BOUND = -1;
ADAPTIVE_BOUND = -2;
WOLFE = 1;
NEWTON_LS = 2;

if nargin < 3
    options = [];
end

n = length(x);
[verbose,maxIter,optTol,progTol,select,stepLen,interp,updateObj,search,nSamples] = myProcessOptions(options,...
    'verbose',1,'maxIter',250,'optTol',1e-5,'progTol',1e-9,...
    'select',CYCLIC,'stepLen',PREVIOUS,'interp',BISECT,'updateObj',0,'search',0,'nSamples',0);

f = obj.objective(x);

if verbose
    fprintf('%10s %15s %15s\n','Cycle','Function','MaxChange');
    fprintf('%10d %15.5e\n',0,f);
end

if search == ADAPTIVE_BOUND
    updateObj = 1;
elseif search == NEWTON_LS
    stepLen = NEWTON;
end

if select == RANDOM_MULTI
    if nSamples == 0
        nSamples = max(1,ceil(log(n)));
    end
elseif select == LIPSCHITZ
   L = obj.maxHessDiag;
   cs = cumsum(L/sum(L));
end

t = ones(n,1);
for i = 1:maxIter
    x_old = x;
    f_old = f;
    done = 1;
    
    for k = 1:n
        
        % Choose coordinate to update
        switch select
            case CYCLIC
                j = k;
                if updateObj
                    g_j = obj.partialFast(j);
                else
                    g_j = obj.partial(x,j);
                end
            case RANDOM
                j = ceil(rand*n);
                if updateObj
                    g_j = obj.partialFast(j);
                else
                    g_j = obj.partial(x,j);
                end
            case LIPSCHITZ
                j = sampleDiscrete_cumsum(cs);
                if updateObj
                    g_j = obj.partialFast(j);
                else
                    g_j = obj.partial(x,j);
                end
            case RANDOM_MULTI
                maxg_j = 0;
                for s = 1:nSamples
                    j = ceil(rand*n);
                    if updateObj
                        g_j = obj.partialFast(j);
                    else
                        g_j = obj.partial(x,j);
                    end
                    if abs(g_j) > maxg_j
                        maxg_j = g_j;
                        maxj = j;
                    end
                end
                j = maxj;
                g_j = maxg_j;
            case GREEDY
                if updateObj
                    g = obj.gradientFast;
                else
                    g = obj.gradient(x);
                end
                [~,j] = max(abs(g));
                g_j = g(j);
                
                if abs(g_j) < optTol
                    done = 1;
                    break;
                end
        end
        
        % Perform line-search and update
        if abs(g_j) > optTol
            done = 0;
            
            % Initialize methods that search for a step-size
            if search >= 0
                % Initialize step size
                if stepLen == NEWTON
                    if updateObj
                        H_jj = obj.HessDiagFast(j);
                    else
                        H_jj = obj.HessDiag(x,j);
                    end
                    t(j) = 1/H_jj;
                end
                
                x_new = x(j) - t(j)*g_j;
                if updateObj
                    obj.setNew(x,j,x_new);
                    f_new = obj.objectiveNew;
                else
                    f_new = obj.objective([x(1:j-1);x_new;x(j+1:end)]);
                end
                
                % Sufficient decrease parameter
                c1 = 1e-4;
            end
            
            switch search
                case BOUND
                    % Fixed step-size line-search
                    t(j) = obj.safeStepSize(j);
                    x_new = x(j) - t(j)*g_j;
                    if updateObj
                        obj.setNew(x,j,x_new);
                        f_new = obj.objectiveNew;
                    else
                        f_new = obj.objective(x);
                    end
                case ADAPTIVE_BOUND
                    t(j) = obj.adaptiveSafeStepSize(j);
                    x_new = x(j) - t(j)*g_j;
                    obj.setNew(x,j,x_new);
                    f_new = obj.objectiveNew;
                case ARMIJO
                    while f_new > f - 1e-4*t(j)*g_j*g_j
                        %if verbose
                        %    fprintf('Backtracking along coordinate %d\n',j);
                        %end
                        if interp == POLY
                            t(j) = polyinterp([0 f -t(j)*g_j*g_j;t(j) f_new sqrt(-1)]);
                        else
                            t(j) = t(j)/2;
                        end
                        x_new = x(j) - t(j)*g_j;
                        if updateObj
                            obj.setNew(x,j,x_new);
                            f_new = obj.objectiveNew;
                        else
                            f_new = obj.objective([x(1:j-1);x_new;x(j+1:end)]);
                        end
                    end
                case WOLFE
                    if updateObj
                        g_new = obj.partialNew(j);
                        c2 = .2; % Precise line-search
                    else
                        g_new = obj.partial([x(1:j-1);x_new;x(j+1:end)],j);
                        c2 = .9; % Imprecise line-search
                    end
                    
                    % Ensure that step size is large enough to bracket minimizer
                    bracketed = 0;
                    wolfeSatisfied = 0;
                    extended = 0;
                    t_prev = 0;
                    f_prev = f;
                    g_prev = g_j;
                    while ~bracketed
                        if f_new > f - 1e-4*t(j)*g_j*g_j || (extended && f_new >= f_prev)
                           %fprintf('Armijo not satisfied or function increased\n');
                           bracket = [0 t(j)];
                           f_bracket = [f f_new];
                           g_bracket = [g_j g_new];
                           bracketed = 1;
                        elseif abs(g_new'*g_j) <= c2*g_j*g_j
                            %fprintf('Wolfe satisfied during bracketing\n');
                            wolfeSatisfied = 1;
                            bracketed = 1;
                        elseif -g_new'*g_j >= 0
                            %fprintf('Armijo satisfied, Wolfe not satisfied and directional derivative is positive\n');
                            bracket = [t_prev t(j)];
                            f_bracket = [f_prev f_new];
                            g_bracket = [g_prev g_new];
                            bracketed = 1;
                        else
                            %fprintf('Armijo satisifed, function decreased, Wolfe not satisfied, directional derivative is negative\n');
                            t(j) = t(j)*2;
                            
                            % Evaluate objective
                            x_new = x(j) - t(j)*g_j;
                            if updateObj
                                obj.setNew(x,j,x_new);
                                f_new = obj.objectiveNew;
                                g_new = obj.partialNew(j);
                            else
                                f_new = obj.objective([x(1:j-1);x_new;x(j+1:end)]);
                                g_new = obj.partial([x(1:j-1);x_new;x(j+1:end)],j);
                            end
                        end
                        extended = extended + 1;
                        %fprintf('Extrapolation %d for variable %d\n',extended,j);
                        t_prev = t(j);
                        f_prev = f_new;
                        g_prev = g_new;
                    end
                    
                    % Refine bracket
                    interpolated = 0;
                    while ~wolfeSatisfied
                        
                        % Compute trial point
                        interpolated = interpolated + 1;
                       %fprintf('Interpolation %d for variable %d\n',interpolated,j);
                        if interp == POLY
                            t(j) = polyinterp([bracket(1) f_bracket(1) -t(j)*g_j*g_bracket(1);bracket(2) f_bracket(2) sqrt(-1)]);
                        else
                            t(j) = (bracket(1)+bracket(2))/2;
                        end
                        
                        % Evaluate objective
                        x_new = x(j) - t(j)*g_j;
                        if updateObj
                            obj.setNew(x,j,x_new);
                            f_new = obj.objectiveNew;
                            g_new = obj.partialNew(j);
                        else
                            f_new = obj.objective([x(1:j-1);x_new;x(j+1:end)]);
                            g_new = obj.partial([x(1:j-1);x_new;x(j+1:end)],j);
                        end
                        
                        [f_LO LOpos] = min(f_bracket);
                        HIpos = -LOpos + 3;
                        f_HI = f_bracket(HIpos);
                        
                        if f_new > f - 1e-4*t(j)*g_j*g_j || f_new >= f_LO
                            %fprintf('Armijo condition not satisfied or new point is higher than best point\n');
                            bracket(HIpos) = t(j);
                            f_bracket(HIpos) = f_new;
                            g_bracket(HIpos) = g_new;
                        elseif abs(g_new'*g_j) <= c2*g_j*g_j
                            %fprintf('Wolfe conditions satisfed\n');
                            wolfeSatisfied = 1;
                        elseif (-g_new'*g_j)*(bracket(HIpos)-bracket(LOpos)) >= 0
                            %fprintf('Old HI is becoming new LO\n');
                            bracket(HIpos) = bracket(LOpos);
                            f_bracket(HIpos) = f_bracket(LOpos);
                            g_bracket(HIpos) = g_bracket(LOpos);
                            
                            bracket(LOpos) = t(j);
                            f_bracket(LOpos) = f_new;
                            g_bracket(LOpos) = g_new;
                        else
                            %fprintf('Old HI is staying the same\n');
                            bracket(LOpos) = t(j);
                            f_bracket(LOpos) = f_new;
                            g_bracket(LOpos) = g_new;
                        end
                        
                        if abs(bracket(1)-bracket(2))*abs(g_j) < progTol
                            %fprintf('Line search failed for variable %d\n',j);
                            break;
                        end
                    end
                case NEWTON_LS
                    x_max = 1e10;
                    x_min = -1e10;
                    if updateObj
                        g_new = obj.partialNew(j);
                    else
                        g_new = obj.partial([x(1:j-1);x_new;x(j+1:end)],j);
                    end
                    
                    while abs(g_new) > optTol
                        x(j) = x_new;
                        f = f_new;
                        if updateObj
                            obj.takeNew;
                        end
                        g_j = g_new;
                        
                       if g_new > 0
                           x_max = x(j);
                       else
                           x_min = x(j);
                       end
                       
                       if updateObj
                           H_jj = obj.HessDiagFast(j);
                       else
                           H_jj = obj.HessDiag(x,j);
                       end
                       t(j) = 1/H_jj;
                        
                       x_new = x(j) - t(j)*g_j;
                       
                       if x_new <= x_min || x_new >= x_max
                           x_new = (x_min+x_max)/2;
                       end
                       
                      if updateObj
                          obj.setNew(x,j,x_new);
                          f_new = obj.objectiveNew;
                          g_new = obj.partialNew(j);
                      else
                          f_new = obj.objective([x(1:j-1);x_new;x(j+1:end)]);
                          g_new = obj.partial([x(1:j-1);x_new;x(j+1:end)],j);
                      end
                      
                    end
            end
            
            % Take step
            x(j) = x_new;
            f = f_new;
            if updateObj
                obj.takeNew;
            end
        end
    end
    
    maxChange = max(abs(x-x_old));
    if verbose
        fprintf('%10d %15.5e %15.5e\n',i,f,maxChange);
    end
    
    if done
        if verbose
            fprintf('Optimality condition on cycle less than optTol\n');
        end
        break;
    end
    
    if abs(f-f_old) < progTol
        if verbose
            fprintf('Change in objective on cycle less than progTol\n');
        end
        break;
    end
    
    if maxChange < progTol
        if verbose
            fprintf('Change in parameter vector on cycle less than progTol\n');
        end
        break;
    end
    
end
