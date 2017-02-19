using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace NN_BP
{
    class MatrixOperation
    {
        private double[,] m_leftMatrix = null;
        private double[,] m_rightMatrix=null;

        private double[,] m_Result=null;

        public static int THREAD_NUM = 64;
        private int m_threadFinishedNum = 0;
        private class MultiplyParameter
        {
            public int r_colStart;
            public int r_colEnd;
            public int l_rowStart;
            public int l_rowEnd;
            public MultiplyParameter(int leftRowStart, int leftRowEnd, int rightColStart, int rightColEnd)
            {
                r_colStart = rightColStart;
                r_colEnd = rightColEnd;
                l_rowStart = leftRowStart;
                l_rowEnd = leftRowEnd;
            }

        }

        public MatrixOperation(double [,]left,double[,] right)
        {
            m_leftMatrix=left;
            m_rightMatrix=right;
        }
        public static void SetThreadNum(int num)
        {
            THREAD_NUM = num;
        }

        private double Sigmoid(double a)
        {
            return 1.0 / (1 + Math.Exp(-a));
        }
        private void  MultiplyMatrix(Object obj)
        {
            MultiplyParameter p=(MultiplyParameter) obj;
            int lrowEnd = p.l_rowEnd;
            int lrowStart = p.l_rowStart;
            int rcolEnd = p.r_colEnd;
            int rcolStart = p.r_colStart;

            
            int dim = m_leftMatrix.GetLength(1);

            for (int i = lrowStart; i < lrowEnd; ++i)
            {
                for (int j = rcolStart; j < rcolEnd; ++j)
                {
                    double temp = 0.0;
                    for (int k = 0; k < dim; ++k)
                    {
                        temp += m_leftMatrix[i, k] * m_rightMatrix[k, j];
                    }
                   // m_Result[i, j] = Sigmoid(temp); //如果不需要对每一个矩阵元素进行任何操作，则把这个Sigmoid(temp)换成temp直接赋值
                    m_Result[i, j] = temp;
                }
            }
            lock(this){
                ++m_threadFinishedNum;
            }
        }

        public void MultiplyMatrixMultiThread()
        {
            Array.Clear(m_Result, 0, m_Result.Length);
            int lRow = m_leftMatrix.GetLength(0);
            int lCol = m_leftMatrix.GetLength(1);
            int rRow = m_rightMatrix.GetLength(0);
            int rCol = m_rightMatrix.GetLength(1);
            double[,] R = new double[lRow, rCol];
            bool splitLeftFlag = true;//按照左边的分割（当然这个得看左边矩阵的行数多，还是右边矩阵的列数多了）
            if (rCol > lRow)
            {
                splitLeftFlag = false;
            }
            m_threadFinishedNum = 0;
            int totalThread=0;
            if (splitLeftFlag)//按行分割左边的矩阵
            {
                if (lRow < THREAD_NUM)
                {
                    totalThread = lRow;
                }
                else
                {
                    totalThread = THREAD_NUM;
                }
                int rowsPerThread = lRow / totalThread;
                int rowsNotAssign = lRow % totalThread;
                int row_start = 0;
                int row_end=0;
                for (int i = 0; i < totalThread; ++i)
                {
                    row_end = row_start + rowsPerThread;
                    if (rowsNotAssign > 0)
                    {
                        ++row_end;  
                        --rowsNotAssign;
                     }
                    MultiplyParameter p = new MultiplyParameter(row_start, row_end, 0, rCol);
                    Thread thread = new Thread(new ParameterizedThreadStart(MultiplyMatrix));
                    row_start = row_end;
                    thread.Start(p);
                }
            }
            else //按列分割右边的矩阵
            {
                if (rCol < THREAD_NUM)
                {
                    totalThread = rCol;
                }
                else
                {
                    totalThread = THREAD_NUM;
                }
                int colsPerThread = rCol / totalThread;
                int colsNotAssign = rCol % totalThread;
                int col_start = 0;
                int col_end = 0;
                for (int i = 0; i < totalThread; ++i)
                {
                    col_end = col_start + colsPerThread;
                    if (colsNotAssign > 0)
                    {
                        ++col_end;      
                        --colsNotAssign;
                    }
                    MultiplyParameter p= new MultiplyParameter(0, lRow, col_start, col_end);
                    Thread thread = new Thread(new ParameterizedThreadStart(MultiplyMatrix));
                    col_start = col_end;
                    thread.Start(p);
                }
            }
            while (m_threadFinishedNum!=totalThread)//直到所有的线程都完成了自己的操作
            {
                Thread.Sleep(10); 
            }

        }

        public double[,] GetResultMatrix()
        {
            return this.m_Result;
        }
        public void SetResultMatrix(ref double[,] result)
        {
            this.m_Result = result;
        }
    }
}
