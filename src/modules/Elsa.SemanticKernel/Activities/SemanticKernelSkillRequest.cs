using System.Net.Http.Headers;
using System.Text.Json.Serialization;
using Elsa.Extensions;
using Elsa.Workflows.Core;
using Elsa.Workflows.Core.Attributes;
using Elsa.Workflows.Core.Contracts;
using Elsa.Workflows.Core.Models;
using JetBrains.Annotations;

namespace Elsa.SemanticKernel;

/// <summary>
/// Invoke a Semantic Kernel skill. 
/// </summary>
[Activity("Elsa", "SemanticKernelSkill", "Invoke a Semantic Kernel skill. ", DisplayName = "Semantic Kernel Skill", Kind = ActivityKind.Task)]
[PublicAPI]
public class SemanticKernelSkill : Activity
{
    [ActivityInput(
        Hint = "System Prompt.",
        UIHint = ActivityInputUIHints.MultiText,
        DefaultValue = new string[0])]
    public string SystemPrompt { get; set; }

    [ActivityInput(
        Hint = "User Input Prompt.",
        UIHint = ActivityInputUIHints.MultiText,
        DefaultValue = new string[0])]
    public string Prompt { get; set; }

    [ActivityInput(
        Hint = "Max retries",
        UIHint = ActivityInputUIHints.SingleLine,
        DefaultValue = 9)]
    public int MaxRetries { get; set; }

    [ActivityInput(
        Hint = "The skill to invoke from the semantic kernel",
        UIHint = ActivityInputUIHints.SingleLine,
        DefaultValue = "PM")]
    public string SkillName { get; set; }

    [ActivityInput(
        Hint = "The function to invoke from the skill",
        UIHint = ActivityInputUIHints.SingleLine,
        DefaultValue = "README")]
    public string FunctionName { get; set; }

    /// <summary>
    /// The output of the skill
    /// </summary>
    [Output(Description = "The output of the skill")]
    public Output<object?> ParsedContent { get; set; } = default!;

    /// <inheritdoc />
    protected override async ValueTask ExecuteAsync(ActivityExecutionContext context)
    {
            var skillName = SkillName;
            var functionName = FunctionName;
            var SystemPrompt = SystemPrompt;
            var prompt = Prompt;
            var result = await ChatCompletion<string>(skillName, functionName, prompt);
            Output = result;

            return Done();
    }

    private async Task<T> ChatCompletion<T>(string skillName, string functionName, string prompt)
    {
        var kernelSettings = KernelSettings.LoadSettings();
        var kernelConfig = new KernelConfig();

        using ILoggerFactory loggerFactory = LoggerFactory.Create(builder =>
        {
            builder
                .SetMinimumLevel(kernelSettings.LogLevel ?? LogLevel.Warning);
        });
        var memoryStore = new QdrantMemoryStore(new QdrantVectorDbClient("http://qdrant", 1536, port: 6333));
        var embedingGeneration = new AzureTextEmbeddingGeneration(kernelSettings.EmbeddingDeploymentOrModelId, kernelSettings.Endpoint, kernelSettings.ApiKey);
        var semanticTextMemory = new SemanticTextMemory(memoryStore, embedingGeneration);

        var kernel = new KernelBuilder()
                            .WithLogger(loggerFactory.CreateLogger<IKernel>())
                            .WithAzureChatCompletionService(kernelSettings.DeploymentOrModelId, kernelSettings.Endpoint, kernelSettings.ApiKey, true, kernelSettings.ServiceId, true)
                            .WithMemory(semanticTextMemory)
                            .WithConfiguration(kernelConfig)
                            .Configure(c => c.SetDefaultHttpRetryConfig(new HttpRetryConfig
                            {
                                MaxRetryCount = MaxRetries,
                                UseExponentialBackoff = true,
                                //  MinRetryDelay = TimeSpan.FromSeconds(2),
                                //  MaxRetryDelay = TimeSpan.FromSeconds(8),
                                MaxTotalRetryTime = TimeSpan.FromSeconds(300),
                                //  RetryableStatusCodes = new[] { HttpStatusCode.TooManyRequests, HttpStatusCode.RequestTimeout },
                                //  RetryableExceptions = new[] { typeof(HttpRequestException) }
                            }))
                            .Build();

        var interestingMemories = kernel.Memory.SearchAsync("ImportedMemories", Prompt, 2);
        var wafContext = "Consider the following contextual snippets:";
        await foreach (var memory in interestingMemories)
        {
            wafContext += $"\n {memory.Metadata.Text}";
        }
        var skillConfig = SemanticFunctionConfig.ForSkillAndFunction(skillName, functionName);
        var function = kernel.CreateSemanticFunction(skillConfig.PromptTemplate, skillConfig.Name, skillConfig.SkillName,
                                                skillConfig.Description, skillConfig.MaxTokens, skillConfig.Temperature,
                                                skillConfig.TopP, skillConfig.PPenalty, skillConfig.FPenalty);

        var context = new ContextVariables();
        context.Set("input", prompt);
        context.Set("wafContext", wafContext);

        var answer = await kernel.RunAsync(context, function).ConfigureAwait(false);
        var result = typeof(T) != typeof(string) ? JsonSerializer.Deserialize<T>(answer.ToString()) : (T)(object)answer.ToString();
        return result;
    }


}