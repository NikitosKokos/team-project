using System.Data.SqlClient;
using System.Threading.Tasks;
using backend.Models;

namespace backend.Repositories
{
    public class VideoRepository
    {
        private readonly string _connectionString;

        public VideoRepository(string connectionString)
        {
            _connectionString = connectionString;
        }

        public async Task LogVideoAsync(StudentInfo studentInfo, int sportID)
        {
            using var connection = new SqlConnection(_connectionString);
            await connection.OpenAsync();

            var query = @"
                INSERT INTO VideoProcessing (VideoName, AthleteID, SportID)
                VALUES (@VideoName, @AthleteID, @SportID)";
            using var command = new SqlCommand(query, connection);
            command.Parameters.AddWithValue("@VideoName", studentInfo.VideoUrl);
            command.Parameters.AddWithValue("@AthleteID", studentInfo.AthleteID);
            command.Parameters.AddWithValue("@SportID", sportID);

            await command.ExecuteNonQueryAsync();
        }
    }
}
