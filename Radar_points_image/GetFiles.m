clear
% 加载测试数据集 Load dataset 
% fulldir target mmwave dataset path YOUR PATH!
fulldir=["E:\mmwave\Data\finaldataset\indoornewperson\normal\2023-03-08-13-54-42",""]; 
% 加载训练数据集
% .mat mmwave path is also okay.
% load("fulldir.mat","fulldir")
% Process item by item getpoint_image
% function getpoint_image To Get point_image 
for i=1:length(fulldir)
    disp(fulldir{i});
    filepart=fileparts(fulldir{i});
    disp(i)
    tic
    getpoint_image([fulldir{i}])
    toc
end
