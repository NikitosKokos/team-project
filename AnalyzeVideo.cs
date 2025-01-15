    using backend.Models;
    using backend.Repositories;
    using backend.Services;
    using Microsoft.Azure.Functions.Worker;
    using Microsoft.Azure.Functions.Worker.Http;
    using Microsoft.Extensions.Logging;
    using System.Net;
    using HttpMultipartParser;

    namespace backend
    {
        public class AnalyzeVideo
        {
            private readonly ILogger<AnalyzeVideo> _logger;
            private readonly BlobService _blobService;
            private readonly VideoRepository _repository;
            private readonly AthleteRepository _athleteRepository;
            private readonly ValidationService _validationService;

            public AnalyzeVideo(ILogger<AnalyzeVideo> logger)
            {
                _logger = logger;

                // Initialize services and repositories
                var sqlConnectionString = Environment.GetEnvironmentVariable("SqlConnectionString") 
                    ?? throw new InvalidOperationException("SqlConnectionString is not configured");
                var blobConnectionString = Environment.GetEnvironmentVariable("BlobConnectionString")
                    ?? throw new InvalidOperationException("BlobConnectionString is not configured");
                var blobContainerName = Environment.GetEnvironmentVariable("BlobContainerName")
                    ?? throw new InvalidOperationException("BlobContainerName is not configured");

                _repository = new VideoRepository(sqlConnectionString);
                _athleteRepository = new AthleteRepository(sqlConnectionString);
                _blobService = new BlobService(blobConnectionString, blobContainerName);
                _validationService = new ValidationService(sqlConnectionString);
            }

            [Function("AnalyzeVideo")]
            public async Task<HttpResponseData> Run(
                [HttpTrigger(AuthorizationLevel.Function, "post")] HttpRequestData req)
            {
                _logger.LogInformation("C# HTTP trigger function processing a video upload request.");

                try
                {
                    // Parse form data
                    var formData = await MultipartFormDataParser.ParseAsync(req.Body);
                    var studentName = formData.GetParameterValue("studentName");
                    var studentClass = formData.GetParameterValue("studentClass");
                    var sportID = int.Parse(formData.GetParameterValue("sportID") ?? throw new ArgumentException("SportID is required."));
                    var file = formData.Files.FirstOrDefault();

                    if (string.IsNullOrWhiteSpace(studentName))
                    {
                        throw new ArgumentException("Student name is required.");
                    }

                    if (file == null || file.Data == null || file.Data.Length == 0)
                    {
                        throw new ArgumentException("Video file is required.");
                    }

                    // Validate sport ID
                    await _validationService.ValidateSportIDAsync(sportID);

                    // Upload the video to Azure Blob Storage
                    var videoUrl = await _blobService.UploadVideoAsync(file.FileName, file.Data);

                    // Get or create the AthleteID
                    var athleteId = await _athleteRepository.GetOrCreateAthleteAsync(studentName, studentClass);

                    // Log video metadata to the database
                    var studentInfo = new StudentInfo
                    {
                        Name = studentName,
                        Class = studentClass,
                        VideoUrl = videoUrl,
                        AthleteID = athleteId
                    };
                    await _repository.LogVideoAsync(studentInfo, sportID);

                    // Return success response
                    var response = req.CreateResponse(HttpStatusCode.OK);
                    await response.WriteStringAsync("Video uploaded and metadata logged successfully.");
                    return response;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing video upload.");
                    var response = req.CreateResponse(HttpStatusCode.BadRequest);
                    await response.WriteStringAsync($"Error: {ex.Message}");
                    return response;
                }
            }
        }
    }
