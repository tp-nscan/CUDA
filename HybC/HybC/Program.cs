using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HybC
{
    class Program
    {
        static void Main(string[] args)
        {
            cuda.DeviceSynchronize();
            HybRunner runner = HybRunner.Cuda("HybC_CUDA.vs2015.dll").SetDistrib(32, 32);
            runner.Wrap(new Program()).Hello();
        }

        [EntryPoint]
        public static void Hello()
        {
            Console.Out.WriteLine ("Hello from GPU");
        }
    }
}
