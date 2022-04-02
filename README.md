## Group DeepMye
## DeepMyeSequence
BADS7604: Deep Learning (HW3)

# Topic: SMS Spam Detection by DL vs ML

## Highlight:
*	ศึกษาการทำนายข้อมูลในรูปแบบของ Text ในรูปแบบข้อความ SMS โดยสามารถแยกข้อมความที่เป็น Spam
*	เปรียบการใช้โมเดล ความแม่นยำ เวลาในการเทรน ระหว่าง Sequential (RNN, LSTM, GRU) และ Traditional Machine Learning (Naive Bayes, Random Forest, Support Vector Machine)
*	Traditional Machine Learning มีความเร็วมากกว่า Sequential Model แต่ความแม่นยำจะต่ำกว่า

## Introduction:
  ทางกลุ่มเขาเราทำการคัดแยก (classification) spam ออกจาก SMS ปกติ โดยทำการ ใช้ Deep Learning ในลักษณะ Sequential Model ได้แก่ Simple RNN, LSTM และ GRU รวมทั้งใช้  Traditional ML ได้แก่ SVM, Naive Bayes, Random Forest และนำผลของแต่ละ Model มาเปรียบเทียบกัน โดยโจทย์ที่เราทำนั้นเป็น Binary Classification จึงใช้ Sigmoid ใน Output Layer ทั้ง 3 Model โดยใช้ ADAM เป็น optimizer และ loss คือ binary cross entropy โดยเราจะ Label เป็น ham คือ SMS ที่ไม่ใช่ spam 

## Data Source:
  ข้อมูลจาก Kaggle คือ text ที่ประกอบไปด้วยข้อความจาก SMS ต่างๆ ข้อมูลที่เราใช้มีลักาณะ imbalance โดยมี spam อยู่ประมาณแค่ 18% จากนั้นได้นำข้อมูลมาแบ่งเป็น 80% เพื่อใช้ในการ Train Model และอีกอย่างละ 10% แบ่งข้อมูลเป็น Validation และ Test สำหรับการทดสอบ Sequential Model และ Machine Learning ก่อนนำข้อมูล SMS ที่เป็นข้อความไปใช้งานเราได้ทำการแบ่งข้อความออกเป็นคำๆ โดยใช้ Word Tokenizerและมีการทำ embedding สำหรับ Sequential Model และนำข้อมูล Text ไปทำ TF-IDF (Term Frequency - Inverse  Document Frequency) และ Transform ข้อมูลก่อนการ Trian ด้วย Machine Learning Model

## Network architecture:
  Simple RNN Model ประกอบไปด้วย
-	Embedding Layer จำนวน 1 Layer (แปลง Word เป็น Vector)
-	RNN Layer ที่มีจำนวน Node = 32 Nodes, Dropout Ratio = 50% จำนวน 1 Layer
-	Output layer ที่ใช้ Sigmoid เป็น Activation Function จำนวน 1 Layer

![image](https://github.com/khwanck/DeepMyeSequence/blob/main/Images/RNN.png)

LSTM Model ประกอบไปด้วย
-	Embedding Layer จำนวน 1 Layer (แปลง Word เป็น Vector)
-	LSTM Layer แบ่งเป็น
1.	LSTM Layer ที่มี 100 Nodes, Dropout ratio = 50% จำนวน 1 Layer
2.	Flatten Layer จำนวน 1 Layer
-	Dense layer ประกอบไปด้วย
1.	Hidden Layer-1 ที่มีจำนวน Node = 200, และใช้ Relu เป็น Activation Function 
2.	Hidden Layer-2 ที่มีจำนวน Node = 100, และใช้ Relu เป็น Activation Function 
3.	Output Layer ที่ใช้ Sigmoid เป็น Activation Function จำนวน 1 Layer

![image](https://github.com/khwanck/DeepMyeSequence/blob/main/Images/LSTM.png)

GRU Model ประกอบไปด้วย
-	Embedding Layer จำนวน 1 Layer (แปลง Word เป็น Vector)
-	GRU Layer แบ่งเป็น
1.	GRU Layer ที่มี 100 Nodes, Dropout ratio = 50% จำนวน 1 Layer
2.	Flatten Layer จำนวน 1 Layer
-	Dense layer ประกอบไปด้วย
1.	Hidden Layer-1 ที่มีจำนวน Node = 200, และใช้ Relu เป็น Activation Function 
2.	Hidden Layer-2 ที่มีจำนวน Node = 100, และใช้ Relu เป็น Activation Function 
3.	Output Layer ที่ใช้ Sigmoid เป็น Activation Function จำนวน 1 Layer

![image](https://github.com/khwanck/DeepMyeSequence/blob/main/Images/GRU.png)

## Training:
  ในการ Train Model เราใช้ข้อมูลจำนวน 80% และ Parameter  batch size = 64, epoch = 20 , validation split = 10% และ Test = 10 % สำหรับ model RNN, LSTM และ GRU   ในส่วนของ ML เราใช้ Data split ออกมาจำนวน 80% และ Test = 10 %

## Results:
   Deep Learning Model ตาม Network Architecture ที่ได้กล่าวไปข้างต้น ได้ Accuracy ประมาณ 97% และเราได้ทำการปรับค่า Parameters ต่างๆ เช่น จำนวน Hidden Layer, จำนวน Node ใน Layer ต่างๆ Dropout Ratio และเปลี่ยน Activation Function พบว่า Accuracy ไม่ได้เปลี่ยนแปลงอย่างมีนัยยะสำคัญ ยังคงได้ Accuracy สูงมากกว่า 90% ทั้งในส่วนของ Train และ Test Data Set และมี Loss ที่ค่อนข้างต่ำอยู่ที่ประมาณ 0.1 โดย RNN จะใช้เวลาในการ Run มากที่สุด และ LSTM และ GRU ใช้เวลาพอๆกัน ในแต่ละ epoch  
   ในส่วนของการใช้ Traditional Machine Learning ในการ Train ข้อมูลนั้นจะใช้เวลาน้อยมากเมื่อเทียบกับการใช้ Deep Learning Model ซึ่งผลลัพธ์ที่ได้ของ Accuracy ค่อนข้างสูงแต่ยังน้อยกว่า DL และค่า Loss จะมีค่ามากกว่าเช่นเดียวกัน

![image](https://github.com/khwanck/DeepMyeSequence/blob/main/Images/loss_accurancy.png)

![image](https://github.com/khwanck/DeepMyeSequence/blob/main/Images/trainingtimes.png)

## Discussion:
  สำหรับความเร็วในการทำงานของ Sequential Model ทั้ง 3 Model นั้นจะพบว่า RNN นั้นทำงานได้เร็วที่สุด รองลงมาคือ GRU และลำดับสุดท้ายคือ LSTM เนื่องจากจำนวน Parameter ที่ต้องทำการ Train นั้น LSTM มีมากที่สุดจึงใช้เวลามากตามไปด้วย
ในด้านประสิทธิภาพ ถ้าเป็นตามทฤษฎี GRU ควรจะต้องมีประสิทธิภาพดีที่สุด ตามมาด้วย LSTM และ RNN ในลำดับสุดท้าย แต่จากที่ทางกลุ่มเราได้ผลออกมานั้น ประสิทธิภาพไม่ต่างกันอย่างมีนัยยะสำคัญ อาจจะเนื่องมาจาก Data นั้นเป็น Data ที่ง่ายและไม่ซับซ้อน ทำให้แค่ Simple RNN ก็สามารถให้ Accuracy ที่ดีได้แล้ว

## Conclusion:
ในด้านความเร็วในการทำงานนั้น Traditional ML ใช้เวลาในการ Run น้อยกว่า Sequential Model แต่อย่างไรก็ตาม ในด้านประสิทธิภาพ ตัวของ Sequential Model ให้ Accuracy ที่สูงกว่า ซึ่งอาจจะเป็นเพราะ Data ที่เรานำมาใช้นั้น เป็นลักษณะของ Time Series ซึ่ง Model ในตระกูลของ RNN จะมีความเหมาะสมกับข้อมูลประเภทนี้มากกว่า
สำหรับความเร็วในการทำงานของ Sequential Model ทั้ง 3 Model นั้นจะพบว่า RNN นั้นทำงานได้เร็วที่สุด รองลงมาคือ GRU และลำดับสุดท้ายคือ LSTM เนื่องจากจำนวน Parameter ที่ต้องทำการ Train นั้น LSTM มีมากที่สุดจึงใช้เวลามากตามไปด้วย
ในด้านประสิทธิภาพ ถ้าเป็นตามทฤษฎี GRU จะต้องมีประสิทธิภาพดีที่สุด ตามมาด้วย LSTM และ RNN ในลำดับสุดท้าย แต่จากที่ทางกลุ่มเราได้ผลออกมานั้น ไม่เห็นความแตกต่างอย่างมีนัยยะสำคัญ


## References:
* https://www.kaggle.com/code/dktalaicha/sms-spam-detection-with-nlp/data
* https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python
* https://www.analyticsvidhya.com/blog/2020/11/understanding-naive-bayes-svm-and-its-implementation-on-spam-sms/
* https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
* https://www.kaggle.com/code/phamdinhkhanh/simple-lstm-for-text-classification-spam-email/notebook

## Team Members

ID   | Responsibility |% Contribute
--------- | ------ | ------
6310422062 | Training RNN | 20%
6310422066 | Training GRU | 20%
6310422067 | Training LSTM | 20%
6310422068 | Training NB & RF | 20%
6310422070 | Training SVM | 20%

## End Credit
โปรเจคนี้เป็นส่วนหนึ่งของวิชา Deep Learning (BADS7604), หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาการวิเคราะห์ธุรกิจและวิทยาการข้อมูล  , สถาบันบัณฑิตพัฒนบริหารศาสตร์ (NIDA)
