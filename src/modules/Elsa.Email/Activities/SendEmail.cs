using System.Collections;
using System.Text;
using System.Text.Json;
using Elsa.Email.Models;
using Elsa.Email.Options;
using Elsa.Email.Services;
using Elsa.Extensions;
using Elsa.Workflows.Core.Attributes;
using Elsa.Workflows.Core.Models;
using Elsa.Workflows.Core.Services;
using Elsa.Workflows.Management.Models;
using Microsoft.Extensions.Options;
using MimeKit;

namespace Elsa.Email.Activities;

/// <summary>
/// Send an email message.
/// </summary>
[Activity("Email", "Send an email message.", Kind = ActivityKind.Task)]
public class SendEmail : Activity
{
    /// <summary>
    /// The sender's email address.
    /// </summary>
    [Input(Description = "The sender's email address.")]
    public Input<string?> From { get; set; }

    [Input(Description = "The recipients email addresses.", UIHint = InputUIHints.MultiText)]
    public Input<ICollection<string>> To { get; set; } = default!;

    [Input(
        Description = "The cc recipient email addresses.",
        UIHint = InputUIHints.MultiText,
        Category = "More")]
    public Input<ICollection<string>> Cc { get; set; } = default!;

    [Input(
        Description = "The Bcc recipients email addresses.",
        UIHint = InputUIHints.MultiText,
        Category = "More")]
    public Input<ICollection<string>> Bcc { get; set; } = default!;

    [Input(Description = "The subject of the email message.")]
    public Input<string?> Subject { get; set; } = default!;

    [Input(
        Description = "The attachments to send with the email message. Can be (an array of) a fully-qualified file path, URL, stream, byte array or instances of EmailAttachment.",
        UIHint = InputUIHints.MultiLine
    )]
    public Input<object?> Attachments { get; set; } = default!;

    [Input(
        Description = "The body of the email message.",
        UIHint = InputUIHints.MultiLine
    )]
    public Input<string?> Body { get; set; } = default!;

    /// <summary>
    /// The activity to execute when an error occurs while trying to send the email.
    /// </summary>
    [Port]
    public IActivity? Error { get; set; }

    protected override async ValueTask ExecuteAsync(ActivityExecutionContext context)
    {
        var cancellationToken = context.CancellationToken;
        var message = new MimeMessage();
        var options = context.GetRequiredService<IOptions<SmtpOptions>>().Value;
        var from = string.IsNullOrWhiteSpace(From.TryGet(context)) ? options.DefaultSender : From.Get(context)!;

        message.Sender = MailboxAddress.Parse(from);
        message.From.Add(MailboxAddress.Parse(from));
        message.Subject = Subject.TryGet(context);

        var bodyBuilder = new BodyBuilder { HtmlBody = Body.TryGet(context) };
        await AddAttachmentsAsync(context, bodyBuilder, cancellationToken);

        message.Body = bodyBuilder.ToMessageBody();

        SetRecipientsEmailAddresses(message.To, To.Get(context));
        SetRecipientsEmailAddresses(message.Cc, Cc.TryGet(context));
        SetRecipientsEmailAddresses(message.Bcc, Bcc.TryGet(context));

        var smtpService = context.GetRequiredService<ISmtpService>();

        try
        {
            await smtpService.SendAsync(message, context.CancellationToken);
            await context.CompleteActivityAsync();
        }
        catch (Exception)
        {
            await context.ScheduleActivityAsync(Error, OnErrorCompletedAsync);
        }
    }

    private async ValueTask OnErrorCompletedAsync(ActivityExecutionContext context, ActivityExecutionContext childContext) => await context.CompleteActivityAsync();

    private async Task AddAttachmentsAsync(ActivityExecutionContext context, BodyBuilder bodyBuilder, CancellationToken cancellationToken)
    {
        var attachments = Attachments.TryGet(context);

        if (attachments == null || attachments is string s && string.IsNullOrWhiteSpace(s))
            return;

        var index = 0;
        var attachmentObjects = InterpretAttachmentsModel(attachments);

        foreach (var attachmentObject in attachmentObjects)
        {
            switch (attachmentObject)
            {
                case Uri url:
                    await AttachOnlineFileAsync(context, bodyBuilder, url, cancellationToken);
                    break;
                case string path when path?.Contains("://") == true:
                    await AttachOnlineFileAsync(context, bodyBuilder, new Uri(path), cancellationToken);
                    break;
                case string path when !string.IsNullOrWhiteSpace(path):
                    await AttachLocalFileAsync(bodyBuilder, path, cancellationToken);
                    break;
                case byte[] bytes:
                {
                    var fileName = $"Attachment-{++index}";
                    bodyBuilder.Attachments.Add(fileName, bytes, ContentType.Parse("application/binary"));
                    break;
                }
                case Stream stream:
                {
                    var fileName = $"Attachment-{++index}";
                    await bodyBuilder.Attachments.AddAsync(fileName, stream, ContentType.Parse("application/binary"), cancellationToken);
                    break;
                }
                case EmailAttachment emailAttachment:
                {
                    var fileName = emailAttachment.FileName ?? $"Attachment-{++index}";
                    var contentType = emailAttachment.ContentType ?? "application/binary";
                    var parsedContentType = ContentType.Parse(contentType);

                    if (emailAttachment.Content is byte[] bytes)
                        bodyBuilder.Attachments.Add(fileName, bytes, parsedContentType);

                    else if (emailAttachment.Content is Stream stream)
                        await bodyBuilder.Attachments.AddAsync(fileName, stream, parsedContentType, cancellationToken);

                    break;
                }
                default:
                {
                    var json = JsonSerializer.Serialize(attachmentObject);
                    var fileName = $"Attachment-{++index}";
                    bodyBuilder.Attachments.Add(fileName, Encoding.UTF8.GetBytes(json), ContentType.Parse("application/json"));
                    break;
                }
            }
        }
    }

    private async Task AttachLocalFileAsync(BodyBuilder bodyBuilder, string path, CancellationToken cancellationToken) => await bodyBuilder.Attachments.AddAsync(path, cancellationToken);

    private async Task AttachOnlineFileAsync(ActivityExecutionContext context, BodyBuilder bodyBuilder, Uri url, CancellationToken cancellationToken)
    {
        var fileName = Path.GetFileName(url.LocalPath);
        var downloader = context.GetRequiredService<IDownloader>();
        var response = await downloader.DownloadAsync(url, cancellationToken);
        var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        var contentType = response.Content.Headers.ContentType?.MediaType ?? "application/binary";
        await bodyBuilder.Attachments.AddAsync(fileName, contentStream, ContentType.Parse(contentType), cancellationToken);
    }

    private IEnumerable InterpretAttachmentsModel(object attachments) => attachments is string text ? new[] { text } : attachments is IEnumerable enumerable ? enumerable : new[] { attachments };

    private void SetRecipientsEmailAddresses(InternetAddressList list, IEnumerable<string>? addresses)
    {
        if (addresses == null)
            return;

        list.AddRange(addresses.Select(MailboxAddress.Parse));
    }
}