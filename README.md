# Set up environment
1. go into workspace **yolov8**
    ```bash
    cd yolov8
    ```
2. set up docker environment
    ```bash
    docker compose up -d --build lab
    ```
3. goto container workspace -> ```/root/code```
4. install requirements
    ```bash
    pip install -r requirements.txt 
    ```

# Train 
1. preprare thermal + RGB pair data
2. create yaml file and set your custom dataset path in ```/root/code/ultralytics/yolo/cfg/xxx.yaml```
3. your custom dataset should put like this
    ```bash
    train_rgb: /root/code/ultralytics/yolo/datasets/LLVIP/RGB/images/train
    val_rgb: /root/code/ultralytics/yolo/datasets/LLVIP/RGB/images/val
    train_ir: /root/code/ultralytics/yolo/datasets/LLVIP/IR/images/train
    val_ir: /root/code/ultralytics/yolo/datasets/LLVIP/IR/images/val
    test_rgb: /root/code/ultralytics/yolo/datasets/LLVIP/RGB/images/val
    test_ir: /root/code/ultralytics/yolo/datasets/LLVIP/IR/images/val
    ```
4. edit ```default.yaml``` 
    ```bash
    data: xxx.yaml
    ```
5. train with your curstom dataset
    ```bash
    python train.py
    ```
# Predict 
1. preprare thermal + RGB pair data
2. edit ```default.yaml``` which in ```/root/code/ultralytics/yolo/cfg/``` 
    * input data

        data type should be **a photo pair** or a **directory** with lot of pair photos
        ```bash
        source_rgb: "/root/code/ultralytics/yolo/datasets/LLVIP/RGB/images/val"
        source_ir: "/root/code/ultralytics/yolo/datasets/LLVIP/IR/images/val"
        ```
    * model 
        ```bash
        model: /root/code/best.pt # path to model file, i.e. yolov8n.pt, yolov8n.yaml
        ```
3. predict 
    ```bash
    python predict_twostream.py
    ```