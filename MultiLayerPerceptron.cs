using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using weka.core;

namespace Z.SemanticSearch
{
    public class MultiLayerPerceptron
    {
        public MultiLayerPerceptron(int number_x, int number_y, List<int> hidden_layers)
        {
            // Set Defaults
            NumberX       = number_x;
            NumberY       = number_y;
            HiddenLayers  = hidden_layers;
            Bias          = 1.0;
            Threshold     = 0;
            Learning_Rate = .1;
            Momentum      = .2;
            Rand          = new Random(GetSeed());
            MaxIterations = 500;
            MinError      = .1; // Min of a 10% error rate.

            InitLayerPerceptrons();
        }

        public int NumberX { get; set; }
        public int NumberY { get; set; }
        public double Bias { get; set; }
        public double Threshold { get; set; }
        public double Learning_Rate { get; set; }
        public double Momentum { get; set; }
        // Each element in List represents the number of nodes in a hidden layer.
        // The Input/Output Layers are created automatically based on the values for X and Y
        public List<int> HiddenLayers { get; set; }
        public List<LayerPerceptron> LayerPerceptrons { get; set; }
        private Random Rand { get; set; } // Random should ultimately come from MLP library.
        public int MaxIterations { get; set; } // Only used with TrainLoop
        // Only used with TrainLoop. This is the min % of data points returning a wrong answer.
        // For multi output answers, if one of the outputs is wrong, then that whole data point is considered wrong.
        public double MinError { get; set; }
        public int Iterations { get; private set; }

        public void InitLayerPerceptrons()
        {
            LayerPerceptrons = new List<LayerPerceptron>(HiddenLayers.Count+2); // hidden.Count + input + output layers

            // Add input layer
            LayerPerceptrons.Add(GetNewLayerPerceptron('i'));

            // Add hidden layers
            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                LayerPerceptrons.Add(GetNewLayerPerceptron(HiddenLayers[i]));
            }

            // Add output layer
            LayerPerceptrons.Add(GetNewLayerPerceptron('o'));
        }

        // Get a new hidden layer
        private LayerPerceptron GetNewLayerPerceptron(int number_perceptrons)
        {
            LayerPerceptron l = new LayerPerceptron(NumberX, NumberY, number_perceptrons);
            l.Bias          = Bias;
            l.Threshold     = Threshold;
            l.Learning_Rate = Learning_Rate;
            l.Momentum      = Momentum;
            l.LayerType     = 'h';
            l.Rand          = Rand;

            l.InitPerceptrons();

            return l;
        }

        // Get a new input or output layer
        private LayerPerceptron GetNewLayerPerceptron(char layer_type)
        {
            LayerPerceptron l = new LayerPerceptron(NumberX, NumberY, layer_type);
            l.Bias            = Bias;
            l.Threshold       = Threshold;
            l.Learning_Rate   = Learning_Rate;
            l.Momentum        = Momentum;
            l.LayerType       = layer_type;
            l.Rand            = Rand;

            l.InitPerceptrons();

            return l;
        }

        private void FeedForward()
        {
            // FeedForward then set current layer output to next layer input for all layers except the output layer.
            for (int i = 0; i < LayerPerceptrons.Count; i++)
            {
                LayerPerceptrons[i].FeedForward();

                if (LayerPerceptrons[i].LayerType != 'o')
                {
                    LayerPerceptrons[i + 1].SetXY(LayerPerceptrons[i].Outputs(), LayerPerceptrons[i].Y);
                }
            }
        }

        private void UpdateWeights()
        {
            SetErrors();
            BackPropogateErrors();
        }

        private void SetErrors()
        {
            // Set errors and weighted error sums for all layers except the input layer.
            for (int i = LayerPerceptrons.Count - 1; i > 0; i--)
            {
                LayerPerceptrons[i].SetErrors();

                if (LayerPerceptrons[i - 1].LayerType != 'i')
                {
                    LayerPerceptrons[i - 1].SetWeightedErrorSum(LayerPerceptrons[i].WeightedErrorSum);
                }
            }
        }

        private void BackPropogateErrors()
        {
            // Backpropogate errors for all layers except the input layer.
            for (int i = LayerPerceptrons.Count - 1; i > 0; i--)
            {
                LayerPerceptrons[i].BackPropogateErrors();
            }
        }

        public void Train(List<double> x, List<double> y)
        {
            LayerPerceptrons[0].SetXY(x, y);
            FeedForward();
            UpdateWeights();
        }

        public List<double> Test(List<double> x, List<double> y)
        {
            LayerPerceptrons[0].SetXY(x, y);

            FeedForward();

            // Return output layer outputs after feeding forward
            return LayerPerceptrons.Last().Outputs();
        }

        public void TrainLoop(List<List<double>> x, List<List<double>> y)
        {
            Iterations = 0;

            while (true)
            {
                int NumberWrong = 0;

                // Train all data points
                for (int i = 0; i < x.Count; i++)
                {
                    Train(x[i], y[i]);
                }

                // Test all data points
                for (int i = 0; i < x.Count; i++)
                {
                    if (HasErrors(Test(x[i], y[i]), y[i]))
                    {
                        NumberWrong++;
                    }
                }

                Iterations++;

                // Stop looping if we have hit the MaxIterations
                if (Iterations >= MaxIterations)
                {
                    break;
                }
                
                // Stop looping if we drop below the minimum error allowed.
                if ((double) NumberWrong/x.Count <= MinError)
                {
                    break;
                }
            }
        }

        private bool HasErrors(List<double> actual, List<double> expected)
        {
            for (int i = 0; i < actual.Count; i++)
            {
                if (actual[i] != expected[i])
                {
                    return true;
                }
            }

            return false;
        }

        public List<double> SaveWeights()
        {
            List<double> w = new List<double>();
 
            for (int i = 0; i < LayerPerceptrons.Count; i++)
            {
                LayerPerceptron lp = LayerPerceptrons[i];

                for (int j = 0; j < lp.Perceptrons.Count; j++)
                {
                    Perceptron p = lp.Perceptrons[j];

                    for (int k = 0; k < p.W.Count; k++)
                    {
                        w.Add(p.W[k]);
                    }
                }
            }
            
            return w;
        }

        public void LoadWeights(List<double> w)
        {
            int count = 0;

            for (int i = 0; i < LayerPerceptrons.Count; i++)
            {
                LayerPerceptron lp = LayerPerceptrons[i];

                for (int j = 0; j < lp.Perceptrons.Count; j++)
                {
                    Perceptron p = lp.Perceptrons[j];

                    for (int k = 0; k < p.W.Count; k++)
                    {
                        p.W[k] = w[count];
                        count++;
                    }
                }
            }
        }

        public new string ToString()
        {
            string s = "";

            for (int i = 0; i < LayerPerceptrons.Count; i++)
            {
                s += "L[" + i.ToString() + "]=\n";
                s += "{\n";
                s += LayerPerceptrons[i].ToString() + "\n";
                s += "}\n";
            }

            return s;
        }

        // The random seed is just seconds from Epoch moded with 1000. Probably not the best. But will work for
        // what I need.
        private int GetSeed()
        {
            DateTime now  = DateTime.Now;
            DateTime then = new DateTime(1970, 1, 1, 0, 0, 0, 0);
            TimeSpan diff = now.ToUniversalTime() - then;

            return ((int) Math.Floor(diff.TotalSeconds))%1000;
        }
    }
}
