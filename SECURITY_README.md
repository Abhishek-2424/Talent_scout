# TalentScout AI - Security Implementation

This document outlines the security measures implemented in the TalentScout AI Interview Assistant to protect sensitive candidate information.

## Security Features

### Data Protection
- **Encryption at Rest**: All sensitive data is encrypted before being stored on disk.
- **Secure Data Storage**: Candidate data is stored in a secure, access-controlled directory.
- **Environment Variables**: Sensitive configuration is stored in environment variables.

### Consent Management
- **Explicit Consent**: Candidates must provide explicit consent before their data is processed.
- **Consent Logging**: All consent actions are logged with timestamps.
- **Consent Expiry**: Consents automatically expire after a configurable period.

### Audit Logging
- **Comprehensive Logging**: All security-relevant actions are logged.
- **Tamper-Evident**: Logs include timestamps and user context.
- **Secure Storage**: Logs are stored securely and protected from unauthorized access.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file with the following variables:
   ```
   SECRET_KEY=your-secret-key-here
   ENCRYPTION_KEY=auto  # Set to 'auto' to generate a new key
   ```

3. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## Security Best Practices

1. **Regular Updates**:
   - Keep all dependencies up to date.
   - Regularly rotate encryption keys.

2. **Access Control**:
   - Restrict access to the application and its data.
   - Use strong, unique passwords.

3. **Monitoring**:
   - Regularly review audit logs.
   - Monitor for suspicious activities.

4. **Data Retention**:
   - Regularly clean up old data.
   - Anonymize or delete data that is no longer needed.

## Troubleshooting

- **Permission Errors**: Ensure the application has write access to the data directory.
- **Encryption Issues**: Verify that the encryption key is set correctly in the `.env` file.
- **Logging Problems**: Check file permissions for the log directory.

For security-related concerns, please contact the development team.
