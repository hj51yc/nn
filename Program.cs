using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace NN_BP
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            double[,] a = new double[,] { { 2, 3, 5 }, { 1, 1, 2 }, { 2, 1, 3 } };
            double[,] b = new double[,] { {3,1},{2,2},{1,1}};
            MatrixOperation op = new MatrixOperation(a, b);
            double [,] c= new double[3,2];
            op.SetResultMatrix(ref c);
            op.MultiplyMatrixMultiThread();

            return;
            */
            DataProcess dp = new DataProcess();
            String trainFile = "D:/data/mnist/train-images.idx3-ubyte";
            String trainLabelFile = "D:/data/mnist/train-labels.idx1-ubyte";
            String testFile = "D:/data/mnist/t10k-images.idx3-ubyte";
            String testLabelFile = "D:/data/mnist/t10k-labels.idx1-ubyte";
            String saveWeightFile = "D:/data/mnist/NN_BP.weights";
            Console.WriteLine("Start to extract data.....");
            dp.ExtractMnistTrainsAndTests(trainFile, trainLabelFile, testFile, testLabelFile);
            dp.NormalizeTheData();
            dp.Shuffle();
            Console.WriteLine("extract data finished!!!");

            int TRAIN_SAMPLE_NUM=dp.trainDatas.Length;
            int TEST_SAMPLE_NUM = dp.testDatas.Length;
            int SAMPLE_DIMENSION = dp.trainDatas[0].Length;
            int HIDEN_NUM = 50;
            int OUTPUT_NUM = 10;
            int BATCH_SIZE = 50;//每十条一个patch

            /*
            double[][] insamples =dp.trainDatas;
            double[][] outsamples = dp.trainLabels;

            double[][] testInSamples = dp.testDatas;
            double[][] testOutSamples = dp.testLabels;
            */
            
            int trainSampleBatchNum = dp.trainDatas.GetLength(0) / BATCH_SIZE;
            if (dp.trainDatas.GetLength(0) % BATCH_SIZE != 0)
            {
                trainSampleBatchNum++;
            }
            int testSampleBatchNum = dp.testDatas.GetLength(0) / BATCH_SIZE;
            if (dp.testDatas.GetLength(0) % BATCH_SIZE != 0)
            {
                testSampleBatchNum++;
            }
            double[][,] trainBatchSamples = new double[trainSampleBatchNum][,];
            double[][,] trainBatchLabels=new double[trainSampleBatchNum][,];
            double[][,] testBatchSamples=new double[testSampleBatchNum][,];
            double[][,] testBatchLabels=new double[testSampleBatchNum][,];
            for (int i = 0; i < trainSampleBatchNum; ++i)
            {
                int start=i*BATCH_SIZE;
                int end=start+BATCH_SIZE;
                if (end > TRAIN_SAMPLE_NUM)
                {
                    end = TRAIN_SAMPLE_NUM;
                }
                double[,] batch=new double[SAMPLE_DIMENSION,end-start];
                double[,] batchLabels = new double[OUTPUT_NUM, end - start];
                int count=0;
                for (int j = start; j < end; ++j)
                {
                    for (int k = 0; k < SAMPLE_DIMENSION; ++k)
                    {
                        batch[k,count]=dp.trainDatas[j][k];
                    }
                    for (int k = 0; k < OUTPUT_NUM; ++k)
                    {
                        batchLabels[k, count] = dp.trainLabels[j][k];
                    }
                    ++count;
                }
                trainBatchSamples[i] = batch;
                trainBatchLabels[i] = batchLabels;
            }
            for (int i = 0; i < testSampleBatchNum; ++i)
            {
                int start = i * BATCH_SIZE;
                int end = start + BATCH_SIZE;
                if (end > TEST_SAMPLE_NUM)
                {
                    end = TEST_SAMPLE_NUM;
                }
                double[,] batch = new double[SAMPLE_DIMENSION, end - start];
                double[,] batchLabels = new double[OUTPUT_NUM, end - start];
                int count = 0;
                for (int j = start; j < end; ++j)
                {
                    for (int k = 0; k < SAMPLE_DIMENSION; ++k)
                    {
                        batch[k, count] = dp.trainDatas[j][k];
                    }
                    for (int k = 0; k < OUTPUT_NUM; ++k)
                    {
                        batchLabels[k, count] = dp.trainLabels[j][k];
                    }
                    ++count;
                }
                testBatchLabels[i] = batchLabels;
                testBatchSamples[i] = batch;
            }


            bool  convergenceFlag=false; //表示结束训练时状态，true表示训练满足阈值要求而结束，false表示round次数达到限制
            int MAX_ROUND=4; //最大的训练次数
            
            double CONVERGENCE_GRADIENT_NORM = 0.0000000001;//当梯度的范数小于这个大小的时候，说明梯度没有明显变化，则停止迭代。
            int THREAD_NUM = 4;

            double curGradientNorm = 0;

            int round=0;

            Stopwatch sw = new Stopwatch();
            sw.Start();
            Stopwatch sw_round = new Stopwatch();
           
            NN_MultiThread neural = new NN_MultiThread(SAMPLE_DIMENSION, HIDEN_NUM, OUTPUT_NUM, THREAD_NUM);
            neural.InitRandomWeights();
            Console.WriteLine("begin find eta:");
            double t0 = neural.DeterminInitT0(trainBatchSamples, trainBatchLabels, 0, 100);
            neural.SetT0(t0);
            Console.WriteLine("training start:");
            double[] costs = new double[MAX_ROUND];
            while (round < MAX_ROUND) 
            {
                sw_round.Start();
                System.Console.WriteLine("Round "+round+" .....");
                neural.Train(trainBatchSamples, trainBatchLabels);
                /*
                curGradientNorm = Math.Sqrt(neural.CalculateGradientSquare());
                System.Console.WriteLine("gradient norm:" + curGradientNorm + " ");
                if (curGradientNorm < CONVERGENCE_GRADIENT_NORM )
                {
                    convergenceFlag = true;
                    break;
                }
                 */
               double cost= neural.Test(testBatchSamples, testBatchLabels);
               costs[round] = cost;
                sw_round.Stop();
                System.Console.WriteLine("Test sample cost:" + cost + " ");
                System.Console.WriteLine("Cost time:" + sw_round.Elapsed + " ");
                ++round;
            } 

            sw.Stop();
            neural.SaveWeightsToFile(saveWeightFile);
            System.Console.WriteLine("Training over!");
            if (!convergenceFlag)
            {
                System.Console.WriteLine("round time overflow , still not visit threshold!");
            }
            else
            {
                System.Console.WriteLine("convergence in "+round+" round ");
            }
            Console.WriteLine("总时间："+sw.Elapsed);
            Console.WriteLine("测量实例得出的总运行时间（毫秒为单位）：" + sw.ElapsedMilliseconds);
            Console.WriteLine("总运行时间(计时器刻度标识)：" + sw.ElapsedTicks);
            Console.WriteLine("计时器是否运行：" + sw.IsRunning.ToString());

            double []results=null;
            bool clsRight = false;
            int totalTestNum = dp.testDatas.Length;
            int clsRightCount = 0;
            for (int i = 0; i < totalTestNum; ++i)
            {
                results = neural.ComputeResultThroughNN(dp.testDatas[i]);
                clsRight = Classify.IsClassifyPassForMnist(dp.testLabels[i], results);
                if (clsRight)
                {
                    ++clsRightCount;
                }
            }
            Console.WriteLine("classfy correct rate:"+(((double)clsRightCount)/totalTestNum));
            Console.Read();
        }
    }
}
