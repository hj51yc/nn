using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;
using System.Diagnostics;
/**
* @author: huangjin (Jeff)
* @email: hj51yc@gmail.com
* Single hidden layer nerual network
**/

namespace NN_BP
{
    class NN_MultiThread
    {

        public const int USE_SIGMOID = 1;
        public const int USE_TANH = 2;

        public int m_ThreadNum; //不能超过64

        private int m_INPUT = 0;//输入神经元个数
        private int m_HIDDEN = 0;//隐藏神经元个数
        private int m_OUTPUT = 0;//输出神经元个数

        private double[] m_WThreshHIDDEN = null;
        private double[] m_WThreshOUTPUT = null;
        private double[] m_GradientThreshHIDDEN = null;//对应权值在某个时候的梯度
        private double[] m_GradientThreshOUTPUT = null;

        private double[,] m_WHI = null;//输入层与隐藏层的权值
        private double[,] m_WOH = null;//隐藏层与输出层的权值
        private double[,] m_GradientWHI = null;//对应权值在某个时候的梯度
        private double[,] m_GradientWOH = null;


        private int m_t = 0;
        private double m_eta0 = 0.0;
        private double m_t0 = 0.0;
        private double m_lamda0 =0.00001;
        //learning rate: lamda0/(t+eta0)

        private int m_threadFinishedNum = 0;
        public void ResetThreadFinishedNum()
        {
            m_threadFinishedNum = 0;
        }
        public void IncThreadFinishedNum()
        {
            m_threadFinishedNum++;
        }
        private class ParamterForGradientV2
        {
            public int start_pos;
            public int end_pos;
            public int event_id;
            public double[,] gradientWOH = null;
            public double[] gradientThreshOUTPUT = null;
            public double[,] gradientWHI = null;
            public  double[] gradientThreshHIDDEN = null;
            public double totalCost;

            public double [,]sampleOutputs=null;
            public double [,]sampleInputs=null;
            public double [,]outOUTPUT=null;
            public double [,]outHIDDEN=null;
            public ParamterForGradientV2(int start, int end)
            {
                start_pos = start;
                end_pos = end;
                //event_id = id;
            }
        }

        public NN_MultiThread(int inputNum, int hidenNum, int outputNum, int THREAD_NUM)
        {
            m_INPUT = inputNum;
            m_HIDDEN = hidenNum;
            m_OUTPUT = outputNum;

            m_WHI = new double[m_HIDDEN, m_INPUT];
            m_WOH = new double[m_OUTPUT, m_HIDDEN];
            m_GradientWHI = new double[m_HIDDEN, m_INPUT];
            m_GradientWOH = new double[m_OUTPUT, m_HIDDEN];

            m_GradientThreshHIDDEN = new double[m_HIDDEN];
            m_GradientThreshOUTPUT = new double[m_OUTPUT];
            m_WThreshHIDDEN = new double[m_HIDDEN];
            m_WThreshOUTPUT = new double[m_OUTPUT];
            m_t = 0;
            m_lamda0 = 0.0005;
            m_eta0 = 1 / m_lamda0;

            m_ThreadNum = THREAD_NUM;
   

        }
        private double Sigmoid(double v)
        {
            return 1.0 / (1.0 + Math.Exp(-v));
        }
        /*
        private double DerivativeSigmoid(double v)
        {
            double sig = Sigmoid(v);
            return sig * (1 - sig);
        }
        */
        private double Tanh(double v)
        {
            double pE = Math.Exp(v);
            double nE = Math.Exp(-v);
            return (pE - nE) / (pE + nE);
        }
        /*
        private double DerivativeTanh(double v)
        {
            double p = Tanh(v);
            return 1 - p * p;
        }
        */

        private void CalculateGradientWork(Object obj)
        {
            ParamterForGradientV2 args = (ParamterForGradientV2)obj;
            int start = args.start_pos;
            int end = args.end_pos;
            double[,] outputs = args.sampleOutputs;
            double[,] inputs = args.sampleInputs;
            double[,] outOUTPUT = args.outOUTPUT;
            double[,] outHIDDEN = args.outHIDDEN;
            double[,] gradientWOH=new double[m_OUTPUT,m_HIDDEN];
            double[,] gradientWHI=new double[m_HIDDEN,m_INPUT];
            double[] gradientThreshHIDDEN = new double[m_HIDDEN];
            double[] gradientThreshOUTPUT = new double[m_OUTPUT];

            Array.Clear(gradientWOH, 0, gradientWOH.Length);
            Array.Clear(gradientWHI, 0, gradientWHI.Length);
            Array.Clear(gradientThreshHIDDEN, 0, gradientThreshHIDDEN.Length);
            Array.Clear(gradientThreshOUTPUT, 0, gradientThreshOUTPUT.Length);

            args.gradientThreshHIDDEN = gradientThreshHIDDEN;
            args.gradientThreshOUTPUT = gradientThreshOUTPUT;
            args.gradientWHI = gradientWHI;
            args.gradientWOH = gradientWOH;

            for (int k = start; k < end; ++k)
            {
                double[] thetaOUTPUT = new double[m_OUTPUT];
                //计算输出层的梯度
                for (int i = 0; i < m_OUTPUT; ++i)
                {
                    double temp = (outputs[i, k] - outOUTPUT[i, k]);
                    thetaOUTPUT[i] = temp * outOUTPUT[i, k] * (1 - outOUTPUT[i, k]);//顺便计算出输出层的theta
                    gradientThreshOUTPUT[i] -= thetaOUTPUT[i];//累计上每个样本造成的误差函数的梯度值，即为多个样本共同训练的梯度值。
                    for (int j = 0; j < m_HIDDEN; ++j)
                    {
                        gradientWOH[i, j] -= thetaOUTPUT[i] * outHIDDEN[j, k]; //输出层与隐藏层的梯度
                    }
                }
                //计算隐藏层的梯度
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    double temp = 0.0;
                    for (int j = 0; j < m_OUTPUT; ++j)
                    {
                        temp += thetaOUTPUT[j] * m_WOH[j, i];
                    }
                    double theta = temp * outHIDDEN[i, k] * (1 - outHIDDEN[i, k]);
                    gradientThreshHIDDEN[i] -= theta;
                    for (int j = 0; j < m_INPUT; ++j)
                    {
                        gradientWHI[i, j] -= theta * inputs[j, k];
                    }
                }
            }
            lock (this)
            {
                IncThreadFinishedNum();
            }
        }

        private void WaitForAllThreadFinished(int totalThread)
        {
            while (m_threadFinishedNum<totalThread)
            {
                Thread.Sleep(20);
            }
        }
        
        //输入输出均以列为方向,每个输入的第一行总是1
        public double CalculateMiniBatchGradient(double[,] inputs, double[,] outputs)
        {
            int sampleDim=inputs.GetLength(0); //samoleDim==m_INPUT
            int sampleNum=inputs.GetLength(1);
            Array.Clear(m_GradientThreshHIDDEN, 0, m_GradientThreshHIDDEN.Length);
            Array.Clear(m_GradientWOH, 0, m_GradientWOH.Length);
            Array.Clear(m_GradientThreshOUTPUT, 0, m_GradientThreshOUTPUT.Length);
            Array.Clear(m_GradientWHI, 0, m_GradientWHI.Length);
          

            //多线程
            double[,] outHIDDEN = new double[m_HIDDEN, sampleNum];
            MatrixOperation op = new MatrixOperation(m_WHI, inputs);
            op.SetResultMatrix(ref outHIDDEN);
            op.MultiplyMatrixMultiThread();
            for (int k = 0; k < sampleNum; ++k)
            {
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    outHIDDEN[i, k] += m_WThreshHIDDEN[i];
                    outHIDDEN[i, k] = Sigmoid(outHIDDEN[i, k]);
                }
            }

            double[,] outOUTPUT = new double[m_OUTPUT, sampleNum];
            op = new MatrixOperation(m_WOH, outHIDDEN);
            op.SetResultMatrix(ref outOUTPUT);
            op.MultiplyMatrixMultiThread();
            double totalCost = 0.0;
            for (int k = 0; k < sampleNum; ++k)
            {
                for (int i = 0; i < m_OUTPUT; ++i)
                {
                    outOUTPUT[i, k] += m_WThreshOUTPUT[i];
                    outOUTPUT[i, k] = Sigmoid(outOUTPUT[i, k]);
                    totalCost += (outOUTPUT[i, k] - outputs[i, k]) * (outOUTPUT[i, k] - outputs[i, k]);
                }
            }
            
            /*
            for (int k = 0; k < sampleNum; ++k)
            {
                double[] thetaOUTPUT = new double[m_OUTPUT];
                //计算输出层的梯度
                for (int i = 0; i < m_OUTPUT; ++i)
                {
                    double temp = (outputs[i, k] - outOUTPUT[i, k]);
                    thetaOUTPUT[i] = temp * outOUTPUT[i, k] * (1 - outOUTPUT[i, k]);//顺便计算出输出层的theta
                    m_GradientThreshOUTPUT[i] -= thetaOUTPUT[i];//累计上每个样本造成的误差函数的梯度值，即为多个样本共同训练的梯度值。
                    for (int j = 0; j < m_HIDDEN; ++j)
                    {
                        m_GradientWOH[i, j] -= thetaOUTPUT[i] * outHIDDEN[j, k]; //输出层与隐藏层的梯度
                    }
                }
                //计算隐藏层的梯度
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    double temp = 0.0;
                    for (int j = 0; j < m_OUTPUT; ++j)
                    {
                        temp += thetaOUTPUT[j] * m_WOH[j, i];
                    }
                    double theta = temp * outHIDDEN[i, k] * (1 - outHIDDEN[i, k]);
                    m_GradientThreshHIDDEN[i] -= theta;
                    for (int j = 0; j < m_INPUT; ++j)
                    {
                        m_GradientWHI[i, j] -= theta * inputs[j, k];
                    }
                }
            }
            */
            
            int totalThread = m_ThreadNum;
            if (m_ThreadNum > sampleNum)
            {
                totalThread = sampleNum;
            }
            int samplePerThread = sampleNum / totalThread;
            int samplesNotAssigned = sampleNum % totalThread;
            int sampleStart = 0;
            int sampleEnd = 0;
            ResetThreadFinishedNum();
            ParamterForGradientV2 []ps=new ParamterForGradientV2[totalThread];
            for (int i = 0; i < totalThread; ++i)
            {
                
                sampleEnd = sampleStart + samplePerThread;
                if (samplesNotAssigned > 0)
                {
                    ++sampleEnd;
                    --samplesNotAssigned;
                }
                ParamterForGradientV2 p = new ParamterForGradientV2(sampleStart, sampleEnd);
                sampleStart = sampleEnd;
                p.outOUTPUT = outOUTPUT;
                p.outHIDDEN = outHIDDEN;
                p.sampleInputs = inputs;
                p.sampleOutputs = outputs;
                ps[i]=p;
                Thread thread = new Thread(new ParameterizedThreadStart(CalculateGradientWork));
                thread.Start(p);
            }
            WaitForAllThreadFinished(totalThread);
            for (int k = 0; k < totalThread; ++k)
            {
                for (int i = 0; i < m_OUTPUT; ++i)
                {
                    m_GradientThreshOUTPUT[i] += ps[k].gradientThreshOUTPUT[i];
                    for (int j = 0; j < m_HIDDEN; ++j)
                    {
                        m_GradientWOH[i, j] += ps[k].gradientWOH[i, j];
                    }
                }
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    m_GradientThreshHIDDEN[i] += ps[k].gradientThreshHIDDEN[i];
                    for (int j = 0; j < m_INPUT; ++j)
                    {
                        m_GradientWHI[i, j] += ps[k].gradientWHI[i,j];
                    }
                }
            }
           

            //单线程
            /*
            double[] outOUTPUT = new double[m_OUTPUT];
            double[] outHIDDEN = new double[m_HIDDEN];
            double[] thetaOUTPUT = new double[m_OUTPUT];
            double totalCost = 0.0;
            for (int k = 0; k < sampleNum; ++k)
            {
                //计算隐藏层的输出
                double sig = 0.0;
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    sig = m_WThreshHIDDEN[i];
                    for (int j = 0; j < m_INPUT; ++j)
                    {
                        sig += m_WHI[i, j] * inputs[j,k];
                    }
                    outHIDDEN[i] = 1.0 / (1.0 + Math.Exp(-sig));
                }
                //计算输出层的输出，并计算梯度
                for (int i = 0; i < m_OUTPUT; ++i)
                {
                    sig = m_WThreshOUTPUT[i];
                    for (int j = 0; j < m_HIDDEN; ++j)
                    {
                        sig += m_WOH[i, j] * outHIDDEN[j];
                    }
                    outOUTPUT[i] = 1.0 / (1.0 + Math.Exp(-sig));
                    double temp = (outputs[i,k] - outOUTPUT[i]);
                    thetaOUTPUT[i] = temp * outOUTPUT[i] * (1 - outOUTPUT[i]);//顺便计算出输出层的theta
                    totalCost += temp * temp; //累计下每个样本每个维度的cost
                    m_GradientThreshOUTPUT[i] -= thetaOUTPUT[i];//累计上每个样本造成的误差函数的梯度值，即为多个样本共同训练的梯度值。
                    for (int j = 0; j < m_HIDDEN; ++j)
                    {
                        m_GradientWOH[i, j] -= thetaOUTPUT[i] * outHIDDEN[j]; //输出层与隐藏层的梯度
                    }
                }

                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    double temp = 0.0;
                    for (int j = 0; j < m_OUTPUT; ++j)
                    {
                        temp += thetaOUTPUT[j] * m_WOH[j, i];
                    }
                    double theta = temp * outHIDDEN[i] * (1 - outHIDDEN[i]);
                    m_GradientThreshHIDDEN[i] -= theta;
                    for (int j = 0; j < m_INPUT; ++j)
                    {
                        m_GradientWHI[i, j] -= theta * inputs[j,k];
                    }
                }
            }
            totalCost /= (2 * sampleNum);
             */
            //求梯度的期望
            ComputeGradientExpectation(sampleNum);
            return totalCost;
        }
        public void TrainOneMiniBatch(double[,] X, double[,] Y,double step) //每个样本为一列
        {
            
            double cost= CalculateMiniBatchGradient(X, Y);
            //根据新的eta更新权值W，W(t+1)=W(t)-(eta0/(lamda*eta0*t+1))W'(t),其中lamda是正规因子，eta0是最初用一部分数据来得倒的最佳的值
            double curGradientNorm = Math.Sqrt(CalculateGradientSquare());
           // System.Console.WriteLine("gradient norm:" + curGradientNorm + " ");
           // System.Console.WriteLine("function cost :" + cost);
            UpdateAllWeights(step);

        }

        public void Train(double[][,] X, double[][,] Y)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            int batchNum = X.GetLength(0);
            for (int i = 0; i < batchNum; ++i)
            {
                ++m_t;
                //Stopwatch sw_round = new Stopwatch();
                //sw_round.Start();
                double step = m_eta0 / (m_t0 + m_t); //learning rate
                TrainOneMiniBatch(X[i], Y[i], step);
               // sw_round.Stop();
               // System.Console.WriteLine("Bacth "+i+" Cost time:" + sw_round.Elapsed + " ");
            }
            sw.Stop();
            System.Console.WriteLine("Train all Cost time:" + sw.Elapsed + " ");
        }

        public void TestOneMiniBatch(double[,] X, double[,] Y, ref double cost)
        {
            int sampleDim = X.GetLength(0); //sampleDim==m_INPUT
            int sampleNum = X.GetLength(1);

            /*
            double[,] inHIDDEN = new double[sampleDim + 1, sampleNum];
            double[,] outHIDDEN = new double[m_HIDDEN, sampleNum];
            double[,] tempWHI = new double[m_HIDDEN, sampleDim + 1];
          

            for (int i = 0; i < sampleNum; ++i)
            {
                for (int j = 0; j < sampleDim; ++j)
                {
                    inHIDDEN[j, i] = X[j, i];
                }
                inHIDDEN[sampleDim, i] = 1; //给隐藏层的输入的最后一行全是1
            }
            for (int i = 0; i < m_HIDDEN; ++i)
            {
                for (int j = 0; j < sampleDim; ++j)
                {
                    tempWHI[i, j] = m_WHI[i, j];
                }
                tempWHI[i, sampleDim] = m_WThreshHIDDEN[i];
            }

            MatrixOperation op = new MatrixOperation(tempWHI, inHIDDEN);
            op.SetResultMatrix(ref outHIDDEN);
            op.MultiplyMatrixAndSigmoidMultiThread();

            double[,] inOUTPUT = new double[m_HIDDEN + 1, sampleNum];
            double[,] outOUTPUT = new double[m_OUTPUT, sampleNum];
            double[,] tempWOH = new double[m_OUTPUT, m_HIDDEN + 1];
           

            for (int i = 0; i < sampleNum; ++i)
            {
                for (int j = 0; j < m_HIDDEN; j++)
                {
                    inOUTPUT[j, i] = outHIDDEN[j, i];
                }
                inOUTPUT[m_HIDDEN, i] = 1;
            }
            for (int i = 0; i < m_OUTPUT; ++i)
            {
                for (int j = 0; j < m_HIDDEN; ++j)
                {
                    tempWOH[i, j] = m_WOH[i, j];
                }
                tempWOH[i, m_HIDDEN] = m_WThreshOUTPUT[i];
            }
            op = new MatrixOperation(tempWOH, inOUTPUT);
            op.SetResultMatrix(ref outOUTPUT);
            op.MultiplyMatrixAndSigmoidMultiThread();
            double temp = 0;
            for (int i = 0; i < sampleNum; ++i)
            {
                for (int j = 0; j < m_OUTPUT; ++j)
                {
                    temp= (outOUTPUT[j, i] - Y[j, i]);
                    cost += temp * temp / 2;
                }
            }
            */
            /*
            double[,] outHIDDEN = new double[m_HIDDEN, sampleNum];
            MatrixOperation op = new MatrixOperation(m_WHI, X);
            op.SetResultMatrix(ref outHIDDEN);
            op.MultiplyMatrixMultiThread();
            for (int k = 0; k < sampleNum; ++k)
            {
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    outHIDDEN[i, k] += m_WThreshHIDDEN[i];
                    outHIDDEN[i, k] = Sigmoid(outHIDDEN[i, k]);
                }
            }
            double[,] outOUTPUT = new double[m_OUTPUT, sampleNum];
            op = new MatrixOperation(m_WOH, outHIDDEN);
            op.SetResultMatrix(ref outOUTPUT);
            op.MultiplyMatrixMultiThread();
            for (int k = 0; k < sampleNum; ++k)
            {
                for (int i = 0; i < m_OUTPUT; ++i)
                {
                    outOUTPUT[i, k] += m_WThreshOUTPUT[i];
                    outOUTPUT[i, k] = Sigmoid(outOUTPUT[i, k]);
                }
            }
            double temp = 0;
            for (int i = 0; i < sampleNum; ++i)
            {
                for (int j = 0; j < m_OUTPUT; ++j)
                {
                    temp = (outOUTPUT[j, i] - Y[j, i]);
                    cost += temp * temp / 2;
                }
            }
             */

            double[] outOUTPUT = new double[m_OUTPUT];
            double[] outHIDDEN = new double[m_HIDDEN];
            double[] thetaOUTPUT = new double[m_OUTPUT];
            double totalCost = 0.0;
            for (int k = 0; k < sampleNum; ++k)
            {
                //计算隐藏层的输出
                double sig = 0.0;
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    sig = m_WThreshHIDDEN[i];
                    for (int j = 0; j < m_INPUT; ++j)
                    {
                        sig += m_WHI[i, j] * X[j, k];
                    }
                    outHIDDEN[i] = 1.0 / (1.0 + Math.Exp(-sig));
                }
                //计算输出层的输出
                for (int i = 0; i < m_OUTPUT; ++i)
                {
                    sig = m_WThreshOUTPUT[i];
                    for (int j = 0; j < m_HIDDEN; ++j)
                    {
                        sig += m_WOH[i, j] * outHIDDEN[j];
                    }
                    outOUTPUT[i] = 1.0 / (1.0 + Math.Exp(-sig));
                    double temp = (Y[i, k] - outOUTPUT[i]);             
                    totalCost += temp * temp; //累计下每个样本每个维度的cost
                }
              
            }
            cost += totalCost;
        }

        public double Test(double[][,] X, double[][,] Y)
        {
            int batchNum = X.GetLength(0);
            double cost = 0.0;
            for (int i = 0; i < batchNum; ++i)
            {
                TestOneMiniBatch(X[i], Y[i], ref cost);
            }
            Console.WriteLine("test error:" + cost);
            return cost;
        }

        public double EvaluteCostByT0(double[][,] X, double[][,] Y, int start,int end,double t0)
        {
            //Console.WriteLine("start evaluate eta:" + t0);
            int batchNum = X.GetLength(0);
            double t=0;
            for (int i = start; i < end && i<batchNum; ++i)
            {
                ++t;
                double step = m_eta0 / (t0+t);
                TrainOneMiniBatch(X[i], Y[i], step);
            }
            double cost = 0.0;
            for (int i = start; i<end && i < batchNum; ++i)
            {
                TestOneMiniBatch(X[i], Y[i], ref cost);
            }
            Console.WriteLine("end evaluate eta:" + t0+"\t cost:"+cost);
            return cost;
        }

        public double DeterminInitT0(double[][,] X, double[][,] Y,int start,int end)
        {
            
            double eta = m_eta0;
            double factor = 2.0;
            double lowT = eta;
            double highT = eta * factor;
            String file = "init_weight";
            SaveWeightsToFile(file);
            double lowCost = EvaluteCostByT0(X, Y, start, end, lowT);
            ReadWeightsFromFile(file);
            double highCost = EvaluteCostByT0(X, Y, start, end, highT);
            if (lowCost<highCost)
            {
                while (lowCost < highCost)
                {
                    highCost = lowCost;
                    highT = lowT;
                    lowT /= factor;
                    ReadWeightsFromFile(file);
                    lowCost = EvaluteCostByT0(X, Y, start, end, lowT);
                }
            }
            else if(lowCost>highCost)
            {
                while (lowCost > highCost)
                {
                    lowCost = highCost;
                    lowT = highT;
                    highT *= factor;
                    ReadWeightsFromFile(file);
                    highCost = EvaluteCostByT0(X, Y, start, end, highT);
                }
            }
            return lowT;
        }

        public void SetT0(double t)
        {
            this.m_t0 = t;
        }
   
        /*
        public void CalculateGradientUseForMultiThread( Object obj)
        {
            ParamterForThread arg = (ParamterForThread)obj;
            int start=arg.start_pos;
            int end=arg.end_pos;
            int event_id = arg.event_id;
            double[] outOUTPUT = new double[m_OUTPUT];
            double[] outHIDDEN = new double[m_HIDDEN];
            double[] thetaOUTPUT = new double[m_OUTPUT];
            double totalCost = 0;
            System.Console.WriteLine(event_id + " thread start.. ,start:"+start+" end:"+end);
            for (int k = start; k < end; ++k)
            {
                //计算隐藏层的输出
                double sig = 0.0;
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    sig = m_WThreshHIDDEN[i];
                    for (int j = 0; j < m_INPUT; ++j)
                    {
                        sig += m_WHI[i, j] * m_testInSamples[k][j];
                    }
                    outHIDDEN[i] = 1.0 / (1.0 + Math.Exp(-sig));
                }
                //计算输出层的输出，并计算梯度
                for (int i = 0; i < m_OUTPUT; ++i)
                {
                    sig = m_WThreshOUTPUT[i];
                    for (int j = 0; j < m_HIDDEN; ++j)
                    {
                        sig += m_WOH[i, j] * outHIDDEN[j];
                    }
                    outOUTPUT[i] = 1.0 / (1.0 + Math.Exp(-sig));
                    double temp = (m_testOutSamples[k][i] - outOUTPUT[i]);
                    thetaOUTPUT[i] = temp * outOUTPUT[i] * (1 - outOUTPUT[i]);//顺便计算出输出层的theta
                     totalCost += temp * temp; //累计下每个样本每个维度的cost
                }
                lock (this)
                {
                    for (int i = 0; i < m_OUTPUT; ++i)
                    {
                   
                            m_GradientThreshOUTPUT[i] -= thetaOUTPUT[i];//累计上每个样本造成的误差函数的梯度值，即为多个样本共同训练的梯度值。
                            for (int j = 0; j < m_HIDDEN; ++j)
                            {
                                m_GradientWOH[i, j] -= thetaOUTPUT[i] * outHIDDEN[j]; //输出层与隐藏层的梯度
                            }
                    }
                }
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    double temp = 0.0;
                    for (int j = 0; j < m_OUTPUT; ++j)
                    {
                        temp += thetaOUTPUT[j] * m_WOH[j, i];
                    }
                    double theta = temp * outHIDDEN[i] * (1 - outHIDDEN[i]);
                    lock (this)
                    {
                        m_GradientThreshHIDDEN[i] -= theta;
                        for (int j = 0; j < m_INPUT; ++j)
                        {
                            m_GradientWHI[i, j] -= theta * m_testInSamples[k][j];
                        }
                    }
                }
            }
            lock (this)
            {
                m_totalCost += totalCost;
            }
            m_Events[event_id].Set();
            System.Console.WriteLine(event_id+" thread finished.. ");
        }

        */
       
        public void CleanTheGradients()
        {
            Array.Clear(m_GradientThreshHIDDEN, 0, m_GradientThreshHIDDEN.Length);
            Array.Clear(m_GradientWOH, 0, m_GradientWOH.Length);
            Array.Clear(m_GradientThreshOUTPUT, 0, m_GradientThreshOUTPUT.Length);
            Array.Clear(m_GradientWHI, 0, m_GradientWHI.Length);
        }

        private double NormValue(int norm, double v)
        {
            if (norm == 1)
            {
                return Math.Abs(v);
            }
            else
            {
                return v * v;
            }
        }

        public void ComputeGradientExpectation(int sampleNum)
        {
            for (int i = 0; i < m_OUTPUT; ++i)
            {
                m_GradientThreshOUTPUT[i] /= sampleNum;
                for (int j = 0; j < m_HIDDEN; ++j)
                {
                    m_GradientWOH[i, j] /= sampleNum;
                }
            }
            for (int i = 0; i < m_HIDDEN; ++i)
            {
                m_GradientThreshHIDDEN[i] /= sampleNum;
                for (int j = 0; j < m_INPUT; ++j)
                {
                    m_GradientWHI[i, j] /= sampleNum;
                }
            }
        }
        public double CalculateGradientSquare()
        {
            double result = 0.0;

            for (int i = 0; i < m_OUTPUT; ++i)
            {
                result += m_GradientThreshOUTPUT[i] * m_GradientThreshOUTPUT[i];
                for (int j = 0; j < m_HIDDEN; ++j)
                {
                    result += m_GradientWOH[i, j] * m_GradientWOH[i, j];
                }
            }
            for (int i = 0; i < m_HIDDEN; ++i)
            {
                result += m_GradientThreshHIDDEN[i] * m_GradientThreshHIDDEN[i];
            }
            for (int j = 0; j < m_INPUT; ++j)
            {
                for (int i = 0; i < m_HIDDEN; ++i)
                {
                    result += m_GradientWHI[i, j] * m_GradientWHI[i, j];
                }
            }
            return result;

        }
        /*
        public double LineSearchAlphaForMultiThread(double max_step, double min_step)//计算梯度下降的最合适的步长
        {
            double theta = 0.1 * CalculateGradientSquare();
            double s = max_step;
            double original_cost = m_totalCost;
            //double original_cost = CalculteCostByStep(0);
            Console.WriteLine("original cost value by calculate:" + original_cost);
            Console.WriteLine("theta:" + theta);
            double temp;
            while (s > min_step)
            {
                int sample_num_per_thread = m_totalSample / m_ThreadNum;
                int sample_start_pos = 0;
                int sample_end_pos = 0;
                ParamterForLineSearch []ps=new ParamterForLineSearch[m_ThreadNum];
                temp = 0;
                for (int i = 0; i < m_ThreadNum; ++i)
                {
                    sample_end_pos = sample_start_pos + sample_num_per_thread;
                    m_Events[i].Reset();
                    ps[i]= new ParamterForLineSearch(s, sample_start_pos,sample_end_pos,i);
                    ThreadPool.QueueUserWorkItem(new WaitCallback(CalculteCostByStepForMultiThread),ps[i]);
                    sample_start_pos=sample_end_pos;
                }
                WaitHandle.WaitAll(m_Events);
                for (int i = 0; i < m_ThreadNum; ++i)
                {
                    temp += ps[i].totalCost;
                }
                 temp/=(2 * m_totalSample); //这一步绝对不能少，否则就不是目标损失度了
                Console.WriteLine("the step:" + s + "\t the cost value is:" + temp);
                if (original_cost - temp >= theta * s)
                {
                    Console.WriteLine("the good function value is:" + temp);
                    return s;
                }

                s /= 2;
            }
            return 0;
        }
         */
        private void UpdateOutputWeights(double step)
        {

            for (int i = 0; i < m_OUTPUT; ++i)
            {
                m_WThreshOUTPUT[i] -= step * m_GradientThreshOUTPUT[i];
                for (int j = 0; j < m_HIDDEN; ++j)
                {
                    m_WOH[i, j] -= step * m_GradientWOH[i, j];
                }
            }
        }
        private void UpdateHidenWeights(double step)
        {

            for (int i = 0; i < m_HIDDEN; ++i)
            {
                m_WThreshHIDDEN[i] -= step * m_GradientThreshHIDDEN[i];
                for (int j = 0; j < m_INPUT; ++j)
                {
                    m_WHI[i, j] -= step * m_GradientWHI[i, j];
                }
            }
        }

        public void UpdateAllWeights(double step)
        {
            UpdateOutputWeights(step);
            UpdateHidenWeights(step);
        }


        public void InitRandomWeights()
        {
            Random rd = new Random();
            for (int i = 0; i < m_OUTPUT; ++i)
            {
                m_WThreshOUTPUT[i] = rd.NextDouble() - 0.5;
                for (int j = 0; j < m_HIDDEN; ++j)
                {
                    m_WOH[i, j] = rd.NextDouble() - 0.5;
                }
            }
            for (int i = 0; i < m_HIDDEN; ++i)
            {
                m_WThreshHIDDEN[i] = rd.NextDouble() - 0.5;
                for (int j = 0; j < m_INPUT; ++j)
                {
                    m_WHI[i, j] = rd.NextDouble() - 0.5;
                }
            }
        }

        public void SaveWeightsToFile(String filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
            BinaryWriter bw = new BinaryWriter(fs);
            for (int i = 0; i < m_OUTPUT; ++i)
            {
                bw.Write(m_WThreshOUTPUT[i]);
                for (int j = 0; j < m_HIDDEN; ++j)
                {
                    bw.Write(m_WOH[i, j]);
                }
            }
            for (int i = 0; i < m_HIDDEN; ++i)
            {
                bw.Write(m_WThreshHIDDEN[i]);
                for (int j = 0; j < m_INPUT; ++j)
                {
                    bw.Write(m_WHI[i, j]);
                }
            }
            bw.Flush();
            bw.Close();
            fs.Close();
        }
        public void ReadWeightsFromFile(String filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read);
            BinaryReader br = new BinaryReader(fs);
            for (int i = 0; i < m_OUTPUT; ++i)
            {
                m_WThreshOUTPUT[i] = br.ReadDouble();
                for (int j = 0; j < m_HIDDEN; ++j)
                {
                    m_WOH[i, j] = br.ReadDouble();
                }
            }
            for (int i = 0; i < m_HIDDEN; ++i)
            {
                m_WThreshHIDDEN[i] = br.ReadDouble();
                for (int j = 0; j < m_INPUT; ++j)
                {
                    m_WHI[i, j] = br.ReadDouble();
                }
            }
            br.Close();
            fs.Close();
        }

        public double[] ComputeResultThroughNN(double[] input)
        {
            if (input.Length != m_INPUT)
            {
                return null;
            }
            double[] results = new double[m_OUTPUT];
            double[] hidden = new double[m_HIDDEN];
            double sig = 0.0;
            for (int i = 0; i < m_HIDDEN; ++i)
            {
                sig = m_WThreshHIDDEN[i];
                for (int j = 0; j < m_INPUT; ++j)
                {
                    sig += m_WHI[i, j] * input[j];
                }
                hidden[i] = 1.0 / (1.0 + Math.Exp(-sig));

            }
            for (int i = 0; i < m_OUTPUT; ++i)
            {
                sig = m_WThreshOUTPUT[i];
                for (int j = 0; j < m_HIDDEN; ++j)
                {
                    sig += m_WOH[i, j] * hidden[j];
                }
                results[i] = 1.0 / (1.0 + Math.Exp(-sig));
            }
            return results;
        }

    }
}
