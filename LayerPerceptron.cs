using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Z.SemanticSearch
{
    public class LayerPerceptron
    {
        // Constructor for hidden layer
        public LayerPerceptron(int number_x, int number_y, int number_perceptrons)
        {
            // Set Defaults
            NumberX           = number_x;
            NumberY           = number_y;
            LayerType         = 'h';
            NumberPerceptrons = number_perceptrons;

            InitDefaults();
        }

        // Constructor for input and output layers
        public LayerPerceptron(int number_x, int number_y, char layer_type)
        {
            // Set Defaults
            NumberX           = number_x;
            NumberY           = number_y;
            LayerType         = layer_type;
            NumberPerceptrons = (LayerType == 'i') ? NumberX : NumberY;

            InitDefaults();
        }

        public int NumberX { get; set; }
        public int NumberY { get; set; }
        public int NumberPerceptrons { get; set; }
        public List<double> X { get; set; }
        public List<double> Y { get; set; }

        public double Bias { get; set; }
        public double Threshold { get; set; }
        public double Learning_Rate { get; set; }
        public double Momentum { get; set; }
        public List<Perceptron> Perceptrons { get; set; }
        public char LayerType { get; set; } // i = input, h = hidden, o = output
        public double WeightedErrorSum { get; set; } // weighted error sum of layer
        public Random Rand { get; set; } // Random should ultimately come from MLP library.
        
        // Init for hidden layer
        private void InitDefaults()
        {
            X                = new List<double>(NumberX);
            Y                = new List<double>(NumberY);
            Bias             = 1.0;
            Threshold        = 0;
            Learning_Rate    = .1;
            Momentum         = .2;
            WeightedErrorSum = 0.0;
            Rand             = new Random();
        }

        public void InitPerceptrons()
        {
            Perceptrons = new List<Perceptron>(NumberPerceptrons);

            for (int i = 0; i < NumberPerceptrons; i++)
            {
                Perceptrons.Add(GetNewPerceptron());
            }
        }

        private Perceptron GetNewPerceptron()
        {
            Perceptron p    = new Perceptron(NumberX);
            p.Bias          = Bias;
            p.Threshold     = Threshold;
            p.Learning_Rate = Learning_Rate;
            p.NodeType      = LayerType;
            p.Momentum      = Momentum;
            p.Rand          = Rand;

            p.InitWeights();

            return p;
        }

        public void SetXY(List<double> x, List<double> y)
        {
            X = x;
            Y = y;

            for (int i = 0; i < NumberPerceptrons; i++)
            {
                List<double> tmpX = new List<double>();
                double tmpY       = 0.0;

                // Set X
                if (LayerType == 'i')
                {
                    // Input Layer will only have one input per perceptron
                    tmpX.Add(X[i]);
                }
                else
                {
                    tmpX.AddRange(X);
                }

                // Set Y
                if (LayerType == 'o')
                {
                    // One output per expected outputs is assigned to each output perceptron.
                    tmpY = Y[i];
                }

                Perceptrons[i].SetXY(tmpX, tmpY);
            }
        }

        public List<double> Outputs()
        {
            List<double> PerceptronOutputs = new List<double>(Perceptrons.Count);

            for (int i = 0; i < NumberPerceptrons; i++)
            {
                PerceptronOutputs.Add(Perceptrons[i].Output());
            }

            return PerceptronOutputs;
        }

        public void FeedForward()
        {
            for (int i = 0; i < NumberPerceptrons; i++)
            {
                Perceptrons[i].SetOutput();
            }
        }

        public void SetErrors()
        {
            WeightedErrorSum = 0.0;

            for (int i = 0; i < NumberPerceptrons; i++)
            {
                Perceptrons[i].SetError();
                WeightedErrorSum += Perceptrons[i].WeightedErrorSum;
            }
        }

        public void SetWeightedErrorSum(double weighted_error_sum)
        {
            for (int i = 0; i < NumberPerceptrons; i++)
            {
                Perceptrons[i].WeightedErrorSum = weighted_error_sum;
            }
        }

        public void BackPropogateErrors()
        {
            for (int i = 0; i < NumberPerceptrons; i++)
            {
                Perceptrons[i].UpdateWeights();
            }
        }

        public new string ToString()
        {
            string s = "";

            for (int i = 0; i < NumberPerceptrons; i++)
            {
                s += "\tP[" + i.ToString() + "]=\n";
                s += "\tWeightedErrorSum: " + WeightedErrorSum.ToString() + "\n";
                s += "\tLayerType: " + LayerType.ToString() + "\n";
                s += Perceptrons[i].ToString() + "\n\n";
            }

            return s;
        }
    }
}
