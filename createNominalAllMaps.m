function handles = createNominalAllMaps(handles)
    disp('createNominalAllMaps started');

    targetFolder = ['C:\Users\BKONG\OneDrive - Xylem Inc\Documents\voaelptest3\Nominal Imgs\'];
    if ~exist(targetFolder,'dir')
        mkdir(targetFolder);
    end
    %loop through all maps except first and last
    for k=2:size(handles.config.defects,1)-1
        
        handles.listbox4Folder.Value = k;
        handles=listbox4Folder(handles);
        %handles=channelSignalCacheAll(handles);

        defectList = handles.config.defects{k};
        defectXPositions = {};
        defectChannels = {};


        %grab all defect x axis positions
        for i=1:length(defectList)
            defectXPositions{i} = [defectList(i).xPositionStart defectList(i).xPositionEnd];
            defectChannels{i} = [defectList(i).channelStart defectList(i).channelEnd];
        end

        counter=1;

        %replace counter< *** with # of nominal imgs to be generated
        while counter<=55        
            %false = no defect interesction, true = defect intersection exists
            boolean=false;

            %initialize random starting x positions and channel for nominal img
            nominalXStart =round(1 + (length(handles.currentMapData.caliperMap)-128-1) .* rand(1,1));
            nominalXEnd = nominalXStart+128-1;
            randChannel = round(1 + (96-1) .* rand(1,1));

            %check if nominal values intersect with existing defects
            for i=1:length(defectList)
                %check if channels intersect with defects
                if (defectChannels{i}(1)<randChannel && defectChannels{i}(2)>randChannel)...
                        || defectChannels{i}(1)==randChannel || defectChannels{i}(2)==randChannel
                    %check if x values intersect with defects
                    if defectXPositions{i}(1)>=nominalXStart && defectXPositions{i}(2)<=nominalXEnd
                        boolean=true;
                    elseif defectXPositions{i}(1)>=nominalXStart && defectXPositions{i}(1)<=nominalXEnd
                        disp(i);
                        boolean=true;
                    elseif defectXPositions{i}(2)>=nominalXStart && defectXPositions{i}(1)<=nominalXEnd
                        disp(i)
                        boolean=true;
                    elseif defectXPositions{i}(1)<=nominalXStart && defectXPositions{i}(2)>=nominalXEnd
                        boolean=true;
                    end
                end
            end

            %if nominal values intersect with existing defects, continue to
            %next loop (forces a loop again without adding to counter to force
            %newly generated nominal values that don't intersect)
            if boolean
                continue;
            end

            %setup channel info
            channelSignalName=strcat('channelSignal_',num2str(randChannel));
            [handles,res] = channelSignalLoad(handles,randChannel);
            channelSignal=handles.channelSignalBuffer.(channelSignalName);
            channelSignal = lineupRingsChannel(randChannel,channelSignal, handles.config.speed{k},1);

            %create nominal img
            defectSignal = channelSignal(1:256,nominalXStart:nominalXEnd);
            fileName = [targetFolder 'Map' num2str(k) '_ch' num2str(randChannel)  '_fr' num2str(nominalXStart) '_to' num2str(nominalXEnd) '.png'];
            defectSignalInt = uint16(32768*(defectSignal+1));
            imwrite(defectSignalInt,fileName);

            counter=counter+1;
        end
    end


end

