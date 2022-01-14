using System.Threading;
using System.Threading.Tasks;
using Elsa.Activities.File.Services;
using Elsa.Services;

namespace Elsa.Activities.File.StartupTasks
{
    public class StartFileSystemWatchers : IStartupTask
    {
        private readonly IFileSystemWatchersStarter _starter;

        public StartFileSystemWatchers(IFileSystemWatchersStarter starter) => _starter = starter;

        public int Order => 2000;

        public Task ExecuteAsync(CancellationToken cancellationToken = default) => _starter.CreateAndAddWatchersAsync(cancellationToken);
    }
}
