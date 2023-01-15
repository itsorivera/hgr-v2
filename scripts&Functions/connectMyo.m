function isConnected = connectMyo()
% Function to connect MYO using MYOMEX library. In the case that the device
% is already connected there is no new action. myoObject is used as a
% global variable

global myoObject

isConnected = 1; % Stablishing successful connection

try
    % Checking existing connection
    isConnected = myoObject.myoData.isStreaming;
    fprintf('Device already connected. \n');
    if isnan(myoObject.myoData.rateEMG)
        fprintf('Problems with rateEMG. \n');
        isConnected = 0;
        
    end
catch
    % In the case that no connection detected
    fprintf('Starting connection. . .\nPlease, wait \n');
    
    try
        % Starting new connection
        myoObject = MyoMex;
        beep
        fprintf('Connection established successfully!!!\n');
        
    catch
        % No connection was possible
        fprintf('Connection was not possible.\nPlease check that MYO Connect is running\n');
        isConnected=0;
        
    end
end

end