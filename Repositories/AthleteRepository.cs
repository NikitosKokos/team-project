using System.Data.SqlClient;
using backend.Models;

namespace backend.Repositories
{
    public class AthleteRepository
    {
        private readonly string _connectionString;

        public AthleteRepository(string connectionString)
        {
            _connectionString = connectionString;
        }

        public async Task<int> GetOrCreateAthleteAsync(string name, string studentClass)
        {
            using var connection = new SqlConnection(_connectionString);
            await connection.OpenAsync();

            // Check if the athlete already exists
            var selectQuery = "SELECT AthleteID FROM Athlete WHERE Name = @Name AND Class = @Class";
            using var selectCommand = new SqlCommand(selectQuery, connection);
            selectCommand.Parameters.AddWithValue("@Name", name);
            selectCommand.Parameters.AddWithValue("@Class", studentClass ?? (object)DBNull.Value);

            var result = await selectCommand.ExecuteScalarAsync();
            if (result != null)
            {
                return Convert.ToInt32(result);
            }

            // Create the athlete if not found
            var insertQuery = "INSERT INTO Athlete (Name, Class) OUTPUT INSERTED.AthleteID VALUES (@Name, @Class)";
            using var insertCommand = new SqlCommand(insertQuery, connection);
            insertCommand.Parameters.AddWithValue("@Name", name);
            insertCommand.Parameters.AddWithValue("@Class", studentClass ?? (object)DBNull.Value);

            var insertResult = await insertCommand.ExecuteScalarAsync();
            return insertResult != null ? (int)insertResult : throw new InvalidOperationException("Failed to insert athlete");
        }
    }
}
