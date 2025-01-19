using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System;
using System.Net.Http;
using Polly;
using Polly.Extensions.Http;

var host = new HostBuilder()
    .ConfigureFunctionsWorkerDefaults()
    .ConfigureServices(services =>
    {
        services.AddHttpClient("DefaultClient")
            .ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler
            {
                ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
            })
            .AddPolicyHandler(GetRetryPolicy());
    })
    .ConfigureLogging(logging =>
    {
        logging.AddConsole();
    })
    .Build();

LogEnvironmentVariables();
host.Run();

static void LogEnvironmentVariables()
{
    var grpcPort = Environment.GetEnvironmentVariable("FUNCTIONS_GRPC_PORT") ?? "7071";
    var runtime = Environment.GetEnvironmentVariable("FUNCTIONS_WORKER_RUNTIME") ?? "dotnet-isolated";
    var port = Environment.GetEnvironmentVariable("WEBSITES_PORT") ?? "80";

    Environment.SetEnvironmentVariable("ASPNETCORE_URLS", $"http://*:{port}");

    Console.WriteLine($"[INFO] FUNCTIONS_GRPC_PORT: {grpcPort}");
    Console.WriteLine($"[INFO] FUNCTIONS_WORKER_RUNTIME: {runtime}");
}


static IAsyncPolicy<HttpResponseMessage> GetRetryPolicy()
{
    return HttpPolicyExtensions
        .HandleTransientHttpError()
        .WaitAndRetryAsync(3, retryAttempt =>
            TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)));
}
