classdef LogisticLossClass < handle
    
    % Variables
    properties
        nInst;
        nVars;
        X;
        y;
        yX;
        yXw;
        yXw_new;
        safeSteps;
        delta;
    end
    
    methods
        
        % Constructor
        function obj = LogisticLossClass(X,y,w)
            obj.nInst = size(X,1);
            obj.nVars = size(X,2);
            obj.X = X;
            obj.y = y;
            obj.yX = -diag(sparse(y))*X;
            if any(w~=0)
                obj.yXw = obj.yX*w;
            else
                obj.yXw = zeros(obj.nInst,1);
            end
            obj.safeSteps = [];
            obj.delta = [];
        end
        
        % Functions that do not use updating
        function f = objective(obj,w)
            f = sum(log(1+exp(obj.yX*w)));
        end
        function p = partial(obj,w,j)
            p = sum(obj.yX(:,j).*(1-1./(1+exp(obj.yX*w))));
        end
        function g = gradient(obj,w)
            g = obj.yX'*(1-1./(1+exp(obj.yX*w)));
        end
        function H = HessDiag(obj,w,j)
            H = sum((1./(1+exp(obj.yX*w))).*(1-1./(1+exp(obj.yX*w))).*(obj.X(:,j).^2));
        end
                
        % Updating functions
        function [obj] = setNew(obj,w,j,w_new)
            obj.yXw_new = obj.yXw + obj.yX(:,j)*(w_new-w(j));
        end
        function [obj] = takeNew(obj)
            obj.yXw = obj.yXw_new;
        end
        
        % Functions that use updating
        function f = objectiveNew(obj)
            f = sum(log(1+exp(obj.yXw_new)));
        end
        function p = partialNew(obj,j)
            p = sum(obj.yX(:,j).*(1-1./(1+exp(obj.yXw_new))));
        end
        function p = partialFast(obj,j)
            p = sum(obj.yX(:,j).*(1-1./(1+exp(obj.yXw))));
        end
        function g = gradientFast(obj)
            g = obj.yX'*(1-1./(1+exp(obj.yXw)));
        end
        function H = HessDiagFast(obj,j)
            H = sum((1./(1+exp(obj.yXw))).*(1-1./(1+exp(obj.yXw))).*(obj.X(:,j).^2));
        end
        
        % Functions calculating global properties
        function Li = maxHessDiag(obj)
            Li = .25*sum(obj.X.^2)';
        end
        function L = maxHessEig(obj)
            L = .25*max(eig(obj.X'*obj.X));
        end
        
        % Functions for specifying step sizes that guarantees descent
        function t = safeStepSize(obj,j)
            if isempty(obj.safeSteps)
                % Initialize
                B = .25*sum(obj.X.^2);
                obj.safeSteps = 1./B;
            end
            t = obj.safeSteps(j);
        end
        
        function t = adaptiveSafeStepSize(obj,j)
            if isempty(obj.safeSteps)
                obj.delta = ones(obj.nVars,1);
            end
            t = 1/sum(obj.F_bound(obj.yXw,obj.delta(j)*abs(obj.X(:,j))).*obj.X(:,j).^2);
            t = min(max(t,-obj.delta(j)),obj.delta(j));
            obj.delta(j) = max(2*abs(t),obj.delta(j)/2);
        end
        function f = F_bound(obj,r,delta)
            f = ones(size(r));
            f = 1./(2 + exp(abs(r)-abs(delta)) + exp(abs(delta)-abs(r)));
            f(abs(r) <= abs(delta)) = .25;
        end
    end
end