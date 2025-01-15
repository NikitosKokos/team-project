using System.Data.SqlClient;
using backend.Models;

namespace backend.Services
{
    public class ValidationService
    {
        private readonly string _connectionString;

        public ValidationService(string connectionString)
        {
            _connectionString = connectionString;
        }

        public async Task ValidateSportIDAsync(int sportID)
        {
            using var connection = new SqlConnection(_connectionString);
            await connection.OpenAsync();

            var query = "SELECT COUNT(1) FROM Sport WHERE SportID = @SportID";
            using var command = new SqlCommand(query, connection);
            command.Parameters.AddWithValue("@SportID", sportID);

            var result = await command.ExecuteScalarAsync();
            var count = result != null ? Convert.ToInt32(result) : throw new InvalidOperationException("Failed to validate SportID");
            if (count == 0)
            {
                throw new ArgumentException($"Invalid SportID: {sportID}");
            }
        }

        public void ValidateStudentInfo(StudentInfo studentInfo)
        {
            if (string.IsNullOrWhiteSpace(studentInfo.Name))
            {
                throw new ArgumentException("Student name is required.");
            }

            if (string.IsNullOrWhiteSpace(studentInfo.VideoUrl))
            {
                throw new ArgumentException("Video URL is required.");
            }
        }
    }
}
