using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace PrediccionInfarto
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            var lector = mlContext.Data.CreateTextLoader(
                new[] {new TextLoader.Column("Descriptores", DataKind.R4, new[] {new TextLoader.Range(0, 12)}),
                new TextLoader.Column("Resultado", DataKind.Bool, 13)},
                separatorChar: ';',
                hasHeader: true);

            const string rutaEntrenamiento = "heart_train.csv";
            const string rutaTest = "heart_test.csv";

            var datosEntrenamiento = lector.Read(rutaEntrenamiento);

            var clfLineal = mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent (labelColumn: "Resultado", featureColumn: "Descriptores");

            var clfLR = mlContext.BinaryClassification.Trainers.LogisticRegression (labelColumn: "Resultado", featureColumn: "Descriptores");

            var clfTrees = 
            mlContext.BinaryClassification.Trainers.FastTree (labelColumn:"Resultado",featureColumn:"Descriptores",numLeaves: 50, numTrees: 30, minDatapointsInLeaves: 20);

            var modelLineal = clfLineal.Fit(datosEntrenamiento);
            var modelLR = clfLR.Fit(datosEntrenamiento);
            var modelTrees = clfTrees.Fit(datosEntrenamiento);

            System.Console.WriteLine("Modelos entrenados correctamente!");

            var datosTest = lector.Read(rutaTest);

            var predLineal = modelLineal.Transform(datosTest);
            var predLR = modelLR.Transform(datosTest);
            var predTrees = modelTrees.Transform(datosTest);

            var metricaLineal = mlContext.BinaryClassification.Evaluate(predLineal, "Resultado");
            var metricaLR = mlContext.BinaryClassification.Evaluate(predLR, "Resultado");
            var metricaTrees = mlContext.BinaryClassification.Evaluate(predTrees, "Resultado");
            
            Console.WriteLine($"Precisión lineal: { metricaLineal.Accuracy:P2}");
            Console.WriteLine($"Precisión regresión logística: { metricaLR.Accuracy:P2}");
            Console.WriteLine($"Precisión árboles de decisión: { metricaTrees.Accuracy:P2}");


        }
    }
}
